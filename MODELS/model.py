import cv2
import torch
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import StreamingResponse
import numpy as np
from io import BytesIO
from PIL import Image

model = torch.hub.load('ultralytics/yolov5', 'yolov5s')  
model.eval()

app = FastAPI()

def process_image(image: Image):
    image = np.array(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    results = model(image)  

    
    img_with_boxes = results.render()[0]  
    return img_with_boxes

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    try:
        image_bytes = await file.read()
        image = Image.open(BytesIO(image_bytes))

        img_with_boxes = process_image(image)

        _, img_encoded = cv2.imencode('.jpg', img_with_boxes)
        img_bytes = img_encoded.tobytes()

        return StreamingResponse(BytesIO(img_bytes), media_type="image/jpeg")

    except Exception as e:
        return {"error": str(e)}
