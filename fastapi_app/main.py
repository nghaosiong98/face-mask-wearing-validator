from typing import List
from fastapi import FastAPI, File, UploadFile, Body
from fastapi.middleware.cors import CORSMiddleware
from model import MaskValidator
import cv2
import numpy as np

app = FastAPI()
model = MaskValidator()

app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*'],
)


@app.get("/validate")
def test_api():
    return {"status": "OK"}


@app.post("/validate")
def predict(images: List[UploadFile] = File(...)):
    response = {
        'results': [],
    }
    for image in images:
        nparr = np.fromstring(image.file.read(), np.uint8)
        raw_image = cv2.imdecode(nparr, cv2.IMREAD_ANYCOLOR)
        result = model.validate_mask(raw_image)
        response['results'].append({
            'filename': image.filename,
            'prediction': result
        })
    return response
