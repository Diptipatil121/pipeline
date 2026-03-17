from fastapi import APIRouter
from app.api import health, pipeline 


api_router = APIRouter(prefix="/api")
api_router.include_router(health.router, tags=["health"])
api_router.include_router(pipeline.router, tags=["pipeline"])