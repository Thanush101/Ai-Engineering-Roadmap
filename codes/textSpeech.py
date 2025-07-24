from dotenv import load_dotenv
from elevenlabs.client import ElevenLabs
from elevenlabs import play
import os

load_dotenv()

elevenlabs = ElevenLabs(
  api_key=os.getenv("ELEVEN_LABS_API_KEY"),
)

audio = elevenlabs.text_to_speech.convert(
    text="So, what's been making you angry lately? Is there something specific that's been triggering those feelings?",
    voice_id="ZUrEGyu8GFMwnHbvLhv2",
    model_id="eleven_multilingual_v2",
    output_format="mp3_44100_128",
)

play(audio)