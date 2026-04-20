import sys
import os
import tempfile
from pathlib import Path

from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from Predictor_models.pipeline.inference import Predictor

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:4200"],
    allow_methods=["POST", "OPTIONS"],
    allow_headers=["*"],
)

predictor = Predictor()


@app.post("/predict")
async def predict(image: UploadFile = File(...)):
    suffix = Path(image.filename).suffix if image.filename else ".jpg"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await image.read())
        tmp_path = tmp.name
    try:
        results = predictor.predict(tmp_path)
    finally:
        os.unlink(tmp_path)
    return results
