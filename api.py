from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
import time

app = FastAPI()

app.add_middleware(CORSMiddleware,
                   allow_origins=["*"],
                   allow_credentials = True,
                   allow_methods=["*"],
                   allow_headers=["*"])

async def fake_video_streamer():
    string = "some fake video bytes"
    for word in string.split():
        time.sleep(1)
        yield word
        
@app.get("/")
async def main():
    return StreamingResponse(fake_video_streamer(),
                             media_type='text/event-stream')