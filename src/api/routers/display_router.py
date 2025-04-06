from fastapi import APIRouter, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

import tensorflow as tf
import pandas as pd
import joblib

router = APIRouter()

templates = Jinja2Templates(directory="templates")

network = tf.keras.models.load_model('src/models/display_predictor.keras')
preprocessor = joblib.load("src/models/preprocessor.pkl")

@router.get("/", response_class=HTMLResponse)
async def show_form(request: Request):
    return templates.TemplateResponse("display_form.html", {"request": request, "title": "Оценка дисплея"})

@router.post("/predict", response_class=HTMLResponse)
async def predict_display(
    request: Request,
    length: float = Form(...),
    width: float = Form(...),
    quality: float = Form(...),
    display_type: str = Form(...),
):
    try:
        size_metric = length * width

        input_dict = {
            "Length": length,
            "Width": width,
            "Quality": quality,
            "Type": display_type,
            "Size_metric": size_metric,
        }

        new_data = pd.DataFrame([input_dict])
        processed = preprocessor.transform(new_data)
        prediction = network.predict(processed, verbose=0)[0]

        defect_level = float(prediction[0])
        quality_score = float(prediction[1])

        return templates.TemplateResponse(
            "display_result.html",
            {
                "request": request,
                "title": "Результаты",
                "length": length,
                "width": width,
                "quality": quality,
                "Type": display_type,
                "defect_level": round(defect_level, 1),
                "quality_score": round(quality_score, 1),
            },
        )
    except Exception as e:
        return templates.TemplateResponse("error.html", {"request": request, "error": str(e), "status_code": 500})