import os
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi import FastAPI, Request
from pydantic import BaseModel
from typing import List
from werkzeug.utils import secure_filename


app = FastAPI()

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Load templates
templates = Jinja2Templates(directory="templates")

# Configuration
UPLOAD_FOLDER = 'static/videos'
ALLOWED_EXTENSIONS = {'mp4'}

class VideoText(BaseModel):
    text: str

class SelectedWords(BaseModel):
    selected_words: List[str]

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.get("/")
def index(request:Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/")
async def upload_video(request: Request, video: UploadFile = File(...)):
    # Check if a file was uploaded
    if video.filename == "":
        return {"error": "No selected file"}

    # Check if the file has a valid extension
    if not allowed_file(video.filename):
        return {"error": "Invalid file format"}

    # Save the uploaded video file
    filename = os.path.join(UPLOAD_FOLDER, secure_filename(video.filename))
    # with open(filename, "wb") as f:
    #     content = await video.read()
    #     f.write(content)
    
    # Perform speech-to-text processing on the video
    text = "We need to make it big"#process_video(filename)
    
    # Split the text into words
    words = text.split()
    
    return templates.TemplateResponse("index.html", {"request": request, "words": words, "video": filename})

@app.post("/process-video")
def process_video_with_words(request: Request, selected_words: SelectedWords, video_filename: str = Form(...)):
    # Process the video with selected words
    #processed_video_filename = process_video(video_filename, selected_words.selected_words)
    
    return templates.TemplateResponse("index.html", {"request": request, "processed_video": video_filename})

def process_video(filename, selected_words):
    # Implement your speech-to-text processing logic here
    # Return the extracted text from the video
    pass

def process_video_with_words(video_filename, selected_words):
    # Implement your video processing logic here
    # Return the filename of the processed video
    pass
