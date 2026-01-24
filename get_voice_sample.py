"""
Download a warm, friendly female voice sample from VCTK corpus
For Chatterbox TTS voice cloning
"""
import os
from datasets import load_dataset
from pydub import AudioSegment
import io

def download_female_voice_sample():
    """Download a young female voice sample (20s-30s age range)"""

    print("📥 Loading VCTK dataset from Hugging Face...")

    # Load dataset
    dataset = load_dataset("CSTR-Edinburgh/vctk", split="train", streaming=True)

    # Filter for young female speakers
    print("🔍 Finding young adult female speakers (20s-30s)...")

    target_speakers = []
    seen_speakers = set()

    for item in dataset:
        speaker_id = item['speaker_id']

        # Skip if we've already seen this speaker
        if speaker_id in seen_speakers:
            continue

        gender = item['gender']
        age = int(item['age'])
        accent = item['accent']
        region = item.get('region', 'Unknown')

        # Filter: Female, 20-30 years old, English accent
        if gender == 'F' and 20 <= age <= 30 and 'English' in accent:
            speaker_info = {
                'speaker_id': speaker_id,
                'age': age,
                'accent': accent,
                'region': region,
                'audio': item['audio'],
                'text': item['text']
            }
            target_speakers.append(speaker_info)
            seen_speakers.add(speaker_id)

            print(f"✅ Found: {speaker_id} - Age {age}, {accent}, {region}")

            # Stop after finding 5 suitable speakers
            if len(target_speakers) >= 5:
                break

    if not target_speakers:
        print("❌ No suitable speakers found")
        return None

    # Use the first speaker (usually good quality)
    selected = target_speakers[0]
    print(f"\n🎯 Selected speaker: {selected['speaker_id']} (Age {selected['age']}, {selected['region']})")
    print(f"📝 Sample text: \"{selected['text']}\"")

    # Get audio data
    audio_data = selected['audio']
    sampling_rate = audio_data['sampling_rate']
    audio_array = audio_data['array']

    # Convert to AudioSegment (for processing)
    print("🎵 Processing audio...")

    # Convert numpy array to bytes
    import numpy as np
    audio_array = np.array(audio_array, dtype=np.float32)

    # Normalize to 16-bit PCM
    audio_16bit = (audio_array * 32767).astype(np.int16)

    # Create audio segment
    audio = AudioSegment(
        audio_16bit.tobytes(),
        frame_rate=sampling_rate,
        sample_width=2,
        channels=1
    )

    # Trim to ~10 seconds (ideal for voice cloning)
    duration_ms = len(audio)
    if duration_ms > 10000:
        audio = audio[:10000]
        print(f"✂️ Trimmed to 10 seconds")
    else:
        print(f"⏱️ Duration: {duration_ms/1000:.1f} seconds")

    # Resample to 24kHz (Chatterbox native rate)
    if sampling_rate != 24000:
        audio = audio.set_frame_rate(24000)
        print(f"🔄 Resampled from {sampling_rate}Hz to 24000Hz")

    # Export as WAV
    output_path = "sarah_voice.wav"
    audio.export(output_path, format="wav")

    print(f"\n✅ Voice sample saved: {output_path}")
    print(f"📊 File size: {os.path.getsize(output_path) / 1024:.1f} KB")
    print(f"⏱️ Duration: {len(audio)/1000:.1f} seconds")
    print(f"🎤 Sample rate: 24000 Hz")

    print("\n" + "="*50)
    print("🎯 NEXT STEPS:")
    print("="*50)
    print("1. Review the voice sample: sarah_voice.wav")
    print("2. If you like it, I'll upload it to the server")
    print("3. Configure ARI bridge to use this voice")
    print("4. Test with a phone call!")

    return output_path

if __name__ == "__main__":
    try:
        download_female_voice_sample()
    except Exception as e:
        print(f"❌ Error: {e}")
        print("\nTrying alternative method...")
        print("You may need to install: pip install datasets pydub soundfile")
