from fastapi import APIRouter
from .routers import clothes_router

router = APIRouter()

router.include_router(clothes_router.router)