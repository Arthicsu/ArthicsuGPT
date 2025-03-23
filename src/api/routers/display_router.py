from fastapi import APIRouter, Request, Form
from fastapi.responses import HTMLResponse

from fastapi.templating import Jinja2Templates
import numpy as np

from src.neural_network import SimpleNeuralNetwork

router = APIRouter()



templates = Jinja2Templates(directory="templates")

network = SimpleNeuralNetwork.load("src/models/neural_network.pkl")

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

        output = network.feedforward(inputs[0])
        predicted_class = np.argmax(output)

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