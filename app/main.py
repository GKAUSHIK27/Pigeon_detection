from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse
import shutil
import os
from detect import process_video  # Assuming this function exists in detect.py
from ultralytics import YOLO

app = FastAPI()

# Load the YOLO model globally so it can be reused
model = YOLO("/weights/best.pt")

@app.post("/upload_video/")
async def upload_video(file: UploadFile = File(...)):
    # Create a directory to store uploaded files if it doesn't exist
    os.makedirs("uploads", exist_ok=True)
    input_video_path = f"uploads/{file.filename}"
    
    # Save the uploaded video to the server
    with open(input_video_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    # Define output video path
    output_video_path = f"uploads/output_{file.filename}"
    
    # Process the video
    process_video(input_video_path, output_video_path, model)

    return FileResponse(output_video_path, media_type='video/mp4')
