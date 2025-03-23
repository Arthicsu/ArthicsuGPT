from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
import uvicorn

from src.api.routers import display_router


app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")
app.include_router(display_router.router)

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0")