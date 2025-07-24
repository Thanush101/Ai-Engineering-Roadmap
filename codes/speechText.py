# Use a pipeline as a high-level helper
from transformers import pipeline
from dotenv import load_dotenv
from langchain.prompts import ChatPromptTemplate
from langchain.chat_models import init_chat_model
from elevenlabs import ElevenLabs
import os
load_dotenv()

audio_file = r"E:\Self Made Projects\Ai-Engineering-Roadmap\resources\about-anger-179423.mp3"

# Speech-to-text using ElevenLabs Scribe model (faster - 13 seconds)
client = ElevenLabs(api_key = os.getenv("ELEVEN_LABS_API_KEY"))

def speech_to_text_elevenlabs(audio_file):
    """Speech-to-text using ElevenLabs Scribe model"""
    try:
        # Correct method call
        with open(audio_file, "rb") as f:
            transcription = client.speech_to_text.convert(
                file=f,
                model_id="scribe_v1",
                tag_audio_events=True,
                diarize=False,
            )
        
        return transcription
        
    except Exception as e:
        print(f"Error: {e}")
        return None

result = speech_to_text_elevenlabs(audio_file)
response = "Transcription:", result.text



# # Text extraaction from audio file using Whisper-small model (slower - 34 seconds)
# pipe = pipeline("automatic-speech-recognition", model="openai/whisper-small", return_timestamps=True)
# result = pipe(audio_file)
# response = result["text"]
# # print(response)

llm = init_chat_model("groq:llama-3.3-70b-versatile", max_tokens=1000, temperature=0.8)

prompt = ChatPromptTemplate.from_messages([
    "system","You are a counsellor that helps people about their problems."
    "Keep your response short and to the point. "
    "Also, keep it as a normal comversation.",
    "human", f"{response}"
])

chain = prompt | llm
llm_response = chain.invoke({})
print(llm_response.content) 
