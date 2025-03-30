from fastapi import APIRouter, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

import numpy as np
import tensorflow as tf

router = APIRouter()


templates = Jinja2Templates(directory="templates")

network = tf.keras.models.load_model('src/models/display_classifier.keras')

@router.get("/", response_class=HTMLResponse)
async def show_form(request: Request):
    return templates.TemplateResponse("display_form.html", {"request": request, "title": "Классификация дисплея"})

@router.post("/classify", response_class=HTMLResponse)
async def classify_display(
    request: Request,
    length: float = Form(...),
    width: float = Form(...),
    quality: float = Form(...),
):
    try:
        inputs = np.array([[length, width, quality]])

        prediction = network.predict(inputs, verbose=0)
        predicted_class = np.argmax(prediction[0])

        display_classes = {
            0: "Маленький, среднего или низкого качества",
            1: "Большой, довольно качественный",
            2: "Большой, плохого качества",
        }
        result = display_classes[predicted_class]

        return templates.TemplateResponse(
            "display_result.html",
            {
                "request": request,
                "title": "Результат классификации дисплея",
                "length": length,
                "width": width,
                "quality": quality,
                "result": result,
            },
        )
    except Exception as e:
        return templates.TemplateResponse("error.html", {"request": request, "error": str(e)})