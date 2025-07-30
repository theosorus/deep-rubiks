from fastapi import APIRouter

from api.routes.cube import cube_router

api_router = APIRouter()


api_router.include_router(cube_router, prefix="/cube")