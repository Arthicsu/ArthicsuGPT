from fastapi import APIRouter
from .routers import instrument_router

router = APIRouter()

router.include_router(instrument_router.router)