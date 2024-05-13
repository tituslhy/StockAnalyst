from io import BytesIO

import chainlit as cl
from chainlit.element import ElementBased
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())

from openai import AsyncOpenAI

from llama_index.core import Settings
from llama_index.llms.openai import OpenAI
# from llama_index.llms.bedrock import Bedrock
from llama_index.core.callbacks import CallbackManager, LlamaDebugHandler

import os
import httpx
import sys
sys.path.append("./src")
from utils import initialize_agent

cl.instrument_openai

client = AsyncOpenAI()

ELEVENLABS_API_KEY = os.environ.get("ELEVENLABS_API_KEY")
ELEVENLABS_VOICE_ID = os.environ.get("ELEVENLABS_VOICE_ID")

if not ELEVENLABS_API_KEY or not ELEVENLABS_VOICE_ID:
    raise ValueError("ELEVENLABS_API_KEY and ELEVENLABS_VOICE_ID must be set")

@cl.step(type="bool")
async def speech_to_text(audio_file):
    response = await client.audio.transcriptions.create(
        model="whisper-1",
        file = audio_file,
        language="en",
    )
    return response.text

@cl.step(type="bool")
async def text_to_speech(text: str, mime_type: str):
    CHUNK_SIZE = 1024
    url = f"https://api.elevenlabs.io/v1/text-to-speech/{ELEVENLABS_VOICE_ID}"
    headers = {
        "Accept": mime_type,
        "Content-Type": "application/json",
        "xi-api-key": ELEVENLABS_API_KEY
    }
    data = {
        "text": text,
        "model_id": "eleven_turbo_v2",
        "voice_settings": {
            "stability": 0.5,
            "similarity_boost": 0.5
        }
    }
    async with httpx.AsyncClient(timeout=25.0) as client:
        response = await client.post(url, json = data, headers = headers)
        response.raise_for_status() #ensures that we are alerted on bad responses
        buffer = BytesIO()
        buffer.name = f"output_audio.{mime_type.split("/")[1]}"
        async for chunk in response.aiter_bytes(chunk_size=CHUNK_SIZE):
            if chunk:
                buffer.write(chunk)
        buffer.seek(0)
        return buffer.name, buffer.read()

@cl.on_chat_start
async def on_chat_start():
    await cl.Avatar(
        name = "Dylan",
        path = "./images/dylan.png"
    ).send()
    llama_debug = LlamaDebugHandler(print_trace_on_end=True)
    Settings.callback_manager = CallbackManager([llama_debug, 
                                                 cl.LlamaIndexCallbackHandler()])
    Settings.llm = OpenAI(model="gpt-4-turbo", temperature=0.1)
    # Settings.llm = Bedrock(
    #     model = "anthropic.claude-3-opus-20240229-v1:0",
    #     aws_access_key_id = os.environ["AWS_ACCESS_KEY"],
    #     aws_secret_access_key = os.environ["AWS_SECRET_ACCESS_KEY"],
    #     aws_region_name = os.environ["AWS_DEFAULT_REGION"]
    # )
    agent = initialize_agent(llm = Settings.llm)
    cl.user_session.set("agent", agent)
    await cl.Message(
        content = "Hi! How can I help you? Simply type your question or press `P` to speak",
        author = "Dylan").send()

@cl.on_message
async def on_message(message: cl.Message):
    agent = cl.user_session.get("agent")
    response_message = await cl.Message(content="", author = "Dylan").send()
    cl.user_session.set("response_message", response_message)
    
    response = await cl.make_async(agent.stream_chat)(message.content)
    response_message = cl.user_session.get("response_message")
    for chunk in response.response_gen:
        await response_message.stream_token(chunk)
    await response_message.update()
    
    if cl.context.session.client_type == "copilot":
        fn = cl.CopilotFunction(name="test", args={"message": message.content,
                                           "response": str(response)})
        await fn.acall()

@cl.on_audio_chunk
async def on_audio_chunk(chunk: cl.AudioChunk):
    if chunk.isStart:
        buffer = BytesIO()
        buffer.name = f"input_audio.{chunk.mimeType.split("/")[1]}"
        cl.user_session.set("audio_buffer", buffer)
        cl.user_session.set("audio_mime_type", chunk.mimeType)
    cl.user_session.get("audio_buffer").write(chunk.data)

@cl.on_audio_end
async def on_audio_end(elements: list[ElementBased]):
    audio_buffer: BytesIO = cl.user_session.get("audio_buffer")
    audio_buffer.seek(0)
    audio_file = audio_buffer.read()
    audio_mime_type: str = cl.user_session.get("audio_mime_type")
    input_audio_el = cl.Audio(
        mime = audio_mime_type, content = audio_file, name=audio_buffer.name
    )
    whisper_input = (audio_buffer.name, audio_file, audio_mime_type)
    transcription = await speech_to_text(whisper_input)
    await cl.Message(
        author="You",
        type = "user_message",
        content = transcription,
        elements = [input_audio_el, *elements]
    ).send()
    
    agent = cl.user_session.get("agent")
    response_message = await cl.Message(content="", author = "Dylan").send()
    cl.user_session.set("response_message", response_message)
    
    response = await cl.make_async(agent.stream_chat)(transcription)
    response_message = cl.user_session.get("response_message")
    for chunk in response.response_gen:
        await response_message.stream_token(chunk)
    output_name, output_audio = await text_to_speech(response_message.content,
                                                     audio_mime_type)
    output_audio_el = cl.Audio(
        name = output_name,
        auto_play = True,
        mime = audio_mime_type,
        content = output_audio,
    )
    response_message.elements = [output_audio_el]
    await response_message.update()
    
    if cl.context.session.client_type == "copilot":
        fn = cl.CopilotFunction(name="test", args={"message": transcription,
                                           "response": str(response)})
        await fn.acall()