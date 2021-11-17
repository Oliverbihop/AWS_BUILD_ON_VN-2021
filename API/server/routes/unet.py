from fastapi import APIRouter, Body
from fastapi.encoders import jsonable_encoder
from server.models.reponse import Response
from server.deep.Unet import Unet


from server.models.unet import (
    UnetSchema
)


router = APIRouter()
Unet = Unet()


@router.post("/", response_description="Predict area")
async def predict_unet(unet: UnetSchema):
    data = jsonable_encoder(unet)
    hashImage = data['hashImage']
    result = Unet.predict(hashImage)
    print(result)
    return Response.ResponseModel(result, "detected successfully")
