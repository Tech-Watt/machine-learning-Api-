from ultralytics import YOLO
import numpy as np 
from PIL import Image
import io 
from fastapi import FastAPI,UploadFile

def load_model():
    modelpath = "best.pt"
    model = YOLO(modelpath)
    return model 

model = load_model()

app = FastAPI()

@app.post('/get_predictions')
async def get_predictions(file:UploadFile):
    image = await file.read()
    image = Image.open(io.BytesIO(image))
    result = model(image)
    names = result[0].names 
    probability = result[0].probs.data.numpy()
    prediction = np.argmax(probability)
    response = {
        'Prediction':names[prediction],
        'Confidence':float(probability[prediction])
    }
    return response
