import sys
import os
import tempfile
from pathlib import Path
from typing import Annotated

from fastapi import FastAPI, File, Form, UploadFile
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

# ── Weights for heuristic clinical risk scorer ─────────────────────────────
# Based on established colorectal cancer (CRC) risk factors.
# Each factor contributes an additive weight; sum is then normalized to [0,1].
_MAX_CLINICAL_SCORE = 1.26  # sum of all max contributions below

def _clinical_risk_score(fields: dict) -> tuple[float, list[str]]:
    score = 0.0
    active_factors: list[str] = []

    def add(factor: str, weight: float, condition: bool) -> None:
        nonlocal score
        if condition:
            score += weight
            active_factors.append(factor)

    add("Sangre en heces (SOF)",    0.25, fields.get("bloodInStool") == "yes")
    add("Rectorragia",              0.20, fields.get("rectorrhagia") == "yes")
    add("Tenesmo",                  0.15, fields.get("tenesmus") == "yes")
    add("Antecedentes familiares",  0.15, fields.get("familyHistory") == "yes")
    add("Diagnóstico previo cáncer",0.10, fields.get("formalCancer") == "yes")
    add("Radioterapia previa",      0.08, fields.get("radiotherapy") == "yes")
    add("Diabetes",                 0.05, fields.get("diabetes") == "yes")

    tobacco = fields.get("tobacco", "never")
    if tobacco == "currently":
        add("Tabaco activo", 0.10, True)
    elif tobacco == "former":
        add("Exfumador", 0.05, True)

    alcohol = fields.get("alcohol", "none")
    alcohol_weight = {"regularly": 0.08, "occasional": 0.04, "former": 0.04}.get(alcohol, 0.0)
    add("Consumo de alcohol", alcohol_weight, alcohol_weight > 0)

    habits = fields.get("intestinalHabits", "normal")
    habits_weight = {
        "alternating": 0.10,
        "incontinence": 0.08,
        "constipation": 0.06,
        "diarrhea": 0.06,
        "others": 0.07,
    }.get(habits, 0.0)
    add("Cambio en hábitos intestinales", habits_weight, habits_weight > 0)

    normalized = min(score / _MAX_CLINICAL_SCORE, 1.0)
    return round(normalized, 4), active_factors


def _risk_level(score: float) -> str:
    if score < 0.30:
        return "bajo"
    if score < 0.60:
        return "moderado"
    if score < 0.80:
        return "alto"
    return "muy_alto"


def _fuse(image_scores: dict, clinical_score: float) -> float:
    # Image risk = highest positive-class signal (pólipos, sangre, inflamación)
    image_risk = max(image_scores.get("polipos", 0),
                     image_scores.get("sangre", 0),
                     image_scores.get("inflamacion", 0))
    fusion = 0.65 * image_risk + 0.35 * clinical_score
    return round(fusion, 4)


@app.post("/predict")
async def predict(
    image: UploadFile = File(...),
    sex: Annotated[str, Form()] = "male",
    tobacco: Annotated[str, Form()] = "never",
    alcohol: Annotated[str, Form()] = "none",
    radiotherapy: Annotated[str, Form()] = "no",
    diabetes: Annotated[str, Form()] = "no",
    formalCancer: Annotated[str, Form()] = "no",
    familyHistory: Annotated[str, Form()] = "no",
    familyHistoryDetails: Annotated[str, Form()] = "",
    bloodInStool: Annotated[str, Form()] = "no",
    rectorrhagia: Annotated[str, Form()] = "no",
    intestinalHabits: Annotated[str, Form()] = "normal",
    tenesmus: Annotated[str, Form()] = "no",
):
    suffix = Path(image.filename).suffix if image.filename else ".jpg"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await image.read())
        tmp_path = tmp.name
    try:
        image_scores = predictor.predict(tmp_path)
    finally:
        os.unlink(tmp_path)

    clinical_fields = dict(
        sex=sex, tobacco=tobacco, alcohol=alcohol,
        radiotherapy=radiotherapy, diabetes=diabetes,
        formalCancer=formalCancer, familyHistory=familyHistory,
        familyHistoryDetails=familyHistoryDetails,
        bloodInStool=bloodInStool, rectorrhagia=rectorrhagia,
        intestinalHabits=intestinalHabits, tenesmus=tenesmus,
    )
    clinical_score, active_factors = _clinical_risk_score(clinical_fields)
    fusion_score = _fuse(image_scores, clinical_score)

    return {
        "image_scores": image_scores,
        "clinical_risk": {
            "score": clinical_score,
            "level": _risk_level(clinical_score),
            "active_factors": active_factors,
        },
        "fusion": {
            "score": fusion_score,
            "level": _risk_level(fusion_score),
            "image_weight": 0.65,
            "clinical_weight": 0.35,
        },
        # flat fields kept for backwards compatibility
        "polipos": image_scores.get("polipos", 0),
        "sangre": image_scores.get("sangre", 0),
        "inflamacion": image_scores.get("inflamacion", 0),
        "negativos": image_scores.get("negativos", 0),
    }
