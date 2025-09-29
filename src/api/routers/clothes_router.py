from fastapi import APIRouter, File, UploadFile, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from tensorflow.keras.models import load_model
from keras.preprocessing import image

import os
from PIL import Image
import numpy as np
import io

router = APIRouter()

templates = Jinja2Templates(directory="templates")

model_path = "src/models/mnist_model.h5"
if os.path.exists(model_path):
    try:
        model = load_model(model_path)
        print("Модель загружена успешно")
    except Exception as e:
        print(f"Ошибка загрузки модели: {e}")
        model = None
else:
    print("Модель не найдена. Запустите train_model.py для обучения модели.")
    model = None

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

@router.get("/", response_class=HTMLResponse)
async def show_form(request: Request):
    return templates.TemplateResponse("clothes_form.html", {"request": request, "title": "Определение одежды по картинке"})

@router.post("/predict", response_class=HTMLResponse)
async def predict_display(
        request: Request,
        file: UploadFile = File(...)
    ):
    try:
        contents = await file.read()
        img = Image.open(io.BytesIO(contents)).convert('L')

        # Изменение размера до 28x28
        img = img.resize((28, 28))

        # Конвертируем в массив numpy
        img_array = image.img_to_array(img)

        # Инвертируем изображение (как в оригинальном примере)
        img_array = 255 - img_array

        # Нормализация
        img_array = img_array / 255.0

        # Добавляем размерности: (1, 28, 28, 1)
        img_array = np.expand_dims(img_array, axis=0)

        prediction = model.predict(img_array)
        predicted_class = np.argmax(prediction)
        confidence = np.max(prediction)

        uploads_dir = "static/uploads"
        os.makedirs(uploads_dir, exist_ok=True)
        img_path = f"{uploads_dir}/{file.filename}"
        img.save(img_path)

        return templates.TemplateResponse("clothes_result.html", {
            "request": request,
            "predicted_class": predicted_class,
            "class_name": class_names[predicted_class],
            "confidence": round(confidence * 100, 2),
            "image_url": f"/{img_path}",
            "all_predictions": [{"class": i, "name": name, "prob": round(prediction[0][i] * 100, 2)}
                                for i, name in enumerate(class_names)]
        })
    except Exception as e:
        return templates.TemplateResponse("error.html", {"request": request, "error": str(e), "status_code": 500})