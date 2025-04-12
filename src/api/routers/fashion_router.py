from fastapi import APIRouter, Request, UploadFile, File
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

import tensorflow as tf
import numpy as np
import uuid


router = APIRouter()
templates = Jinja2Templates(directory="templates")

model = tf.keras.models.load_model('src/models/fashion_model_v3.keras')
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']


@router.get("/", response_class=HTMLResponse)
async def show_form(request: Request):
    return templates.TemplateResponse("display_form.html", {"request": request, "title": "Распознавание одежды"})


@router.post("/predict", response_class=HTMLResponse)
async def predict_display(
        request: Request,
        file: UploadFile = File(...)
):
    try:
        temp_filename = f"assets/temp/{uuid.uuid4()}.png"
        temp_filepath = f"static/{temp_filename}"

        with open(temp_filepath, "wb") as buffer:
            buffer.write(await file.read())

        img = tf.keras.preprocessing.image.load_img(
            temp_filepath,
            target_size=(28, 28),
            color_mode='grayscale'
        )

        x = tf.keras.preprocessing.image.img_to_array(img)
        x = x.reshape(1, 784).astype('float32')
        x = (255 - x) / 255

        pred = model.predict(x)
        pred_class = np.argmax(pred)

        return templates.TemplateResponse(
            "display_result.html",{
                "request": request,
                "title": "Результаты",
                "predicted_class": class_names[pred_class],
                "ver": round(float(np.max(pred[0])), 4),
                "image_url": f"/static/{temp_filename}"
            },
        )
    except Exception as e:
        return templates.TemplateResponse("error.html", {"request": request, "error": str(e), "status_code": 500})