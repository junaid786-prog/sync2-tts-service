"""
Create a voice sample using Google TTS for Chatterbox voice cloning
"""
from gtts import gTTS
import os

# Sample text for voice cloning (medical context)
text = """
Hello, this is Sarah from the medical office.
I'm calling to confirm your appointment with Doctor Johnson tomorrow at ten thirty a.m.
Please make sure to bring your insurance card and arrive fifteen minutes early.
If you need to reschedule, please call us as soon as possible.
Thank you and have a great day.
"""

print("Generating voice sample with Google TTS...")
tts = gTTS(text=text, lang='en', slow=False)

# Save as MP3 first
output_file = "sarah_voice_gtts.mp3"
tts.save(output_file)
print(f"Voice sample saved: {output_file}")
print(f"File size: {os.path.getsize(output_file) / 1024:.1f} KB")

print("\nNow convert to WAV format using ffmpeg...")
print("Run: ffmpeg -i sarah_voice_gtts.mp3 -ar 24000 -ac 1 sarah_voice.wav")
