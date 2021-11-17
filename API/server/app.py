from fastapi import FastAPI

from server.routes.area import router as AreaRoutes
from server.routes.unet import router as UnetRoutes



from starlette.middleware.cors import CORSMiddleware




app = FastAPI()


app.include_router(AreaRoutes, tags=["Area"], prefix="/area")
app.include_router(UnetRoutes, tags=["Unet"], prefix="/unet")


@app.get("/", tags=["Root"])
async def read_root():
    return {"message": "Welcome to this fantastic app!"}




app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["POST","PUT","DELETE", "OPTION", "GET"],
    allow_headers=["*"],
)