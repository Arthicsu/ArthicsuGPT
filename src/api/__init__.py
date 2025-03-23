from fastapi import APIRouter
from .routers import display_router

router = APIRouter()

router.include_router(display_router.router)