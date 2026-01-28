"""
Test client for CSM-1B TTS Server
"""

import asyncio
import json
import time
import wave
import io

import websockets


async def test_tts(
    text: str = "Hello, this is a test of the CSM speech synthesis model.",
    server_url: str = "ws://localhost:8765/tts/stream",
    output_file: str = "test_output.wav"
):
    """Test TTS synthesis"""
    print(f"Connecting to {server_url}...")

    async with websockets.connect(server_url) as ws:
        # Send request
        request = {
            "text": text,
            "format": "wav",  # or "ulaw" for telephony
            "temperature": 0.7,
            "speed": 1.0
        }

        print(f"Sending: '{text}'")
        start_time = time.time()

        await ws.send(json.dumps(request))

        # Receive audio chunks
        audio_data = b""
        while True:
            chunk = await ws.recv()
            if chunk == b"__END__":
                break
            audio_data += chunk

        elapsed = time.time() - start_time
        print(f"Received {len(audio_data)} bytes in {elapsed:.2f}s")

        # Save to file
        with open(output_file, "wb") as f:
            f.write(audio_data)

        print(f"Saved to {output_file}")


async def test_streaming(
    text: str = "This is a streaming test with multiple sentences. The audio should arrive in chunks.",
    server_url: str = "ws://localhost:8765/tts/realtime"
):
    """Test streaming TTS"""
    print(f"Testing streaming at {server_url}...")

    async with websockets.connect(server_url) as ws:
        request = {"text": text}
        await ws.send(json.dumps(request))

        chunk_count = 0
        total_bytes = 0

        while True:
            chunk = await ws.recv()
            if chunk == b"__END__":
                break
            chunk_count += 1
            total_bytes += len(chunk)
            print(f"Chunk {chunk_count}: {len(chunk)} bytes")

        print(f"Total: {chunk_count} chunks, {total_bytes} bytes")


async def test_health(server_url: str = "ws://localhost:8765/health"):
    """Test health endpoint"""
    async with websockets.connect(server_url) as ws:
        response = await ws.recv()
        print(f"Health: {response}")


async def benchmark(
    texts: list = None,
    server_url: str = "ws://localhost:8765/tts/stream"
):
    """Benchmark TTS performance"""
    if texts is None:
        texts = [
            "Hello.",
            "This is a short sentence.",
            "This is a medium length sentence with more words.",
            "This is a longer sentence that contains many words and should take more time to synthesize into speech audio.",
        ]

    print("Running benchmark...")
    results = []

    for text in texts:
        async with websockets.connect(server_url) as ws:
            request = {"text": text, "format": "ulaw"}

            start = time.time()
            await ws.send(json.dumps(request))

            audio_data = b""
            while True:
                chunk = await ws.recv()
                if chunk == b"__END__":
                    break
                audio_data += chunk

            elapsed = time.time() - start
            audio_duration = len(audio_data) / 8000  # ulaw at 8kHz

            results.append({
                "text_len": len(text),
                "audio_bytes": len(audio_data),
                "audio_duration": audio_duration,
                "latency": elapsed,
                "rtf": elapsed / audio_duration if audio_duration > 0 else 0
            })

            print(f"Text: {len(text)} chars -> {audio_duration:.2f}s audio in {elapsed:.2f}s (RTF: {results[-1]['rtf']:.2f})")

    avg_rtf = sum(r["rtf"] for r in results) / len(results)
    print(f"\nAverage RTF: {avg_rtf:.2f} (< 1.0 means faster than real-time)")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        mode = sys.argv[1]
        if mode == "stream":
            asyncio.run(test_streaming())
        elif mode == "health":
            asyncio.run(test_health())
        elif mode == "bench":
            asyncio.run(benchmark())
        else:
            asyncio.run(test_tts(text=mode))
    else:
        asyncio.run(test_tts())
