from fastapi import APIRouter, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

import pandas as pd
import joblib

router = APIRouter()

templates = Jinja2Templates(directory="templates")

model = joblib.load("src/models/instrument_pickle_file.pkl")

@router.get("/", response_class=HTMLResponse)
async def show_form(request: Request):
    return templates.TemplateResponse("instrument_form.html", {"request": request, "title": "Определение музыкального инструмента"})

@router.post("/predict", response_class=HTMLResponse)
async def predict_display(
    request: Request,
    height: float = Form(...),
    loudness: float = Form(...),
):
    try:
        new_data = pd.DataFrame([[height, loudness]], columns=["height", "loudness"])
        prediction = model.predict(new_data)[0]
        input_dict = {
            0: "Барабан",
            1: "Флейта",
            2: "Гитара",
        }

        return templates.TemplateResponse(
            "instrument_result.html",
            {
                "request": request,
                "title": "Результаты",
                "height": height,
                "loudness": loudness,
                "instrument": input_dict[prediction],
            },
        )
    except Exception as e:
        return templates.TemplateResponse("error.html", {"request": request, "error": str(e), "status_code": 500})