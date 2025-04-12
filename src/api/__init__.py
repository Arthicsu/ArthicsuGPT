from fastapi import APIRouter
from .routers import fashion_router

router = APIRouter()

router.include_router(fashion_router.router)