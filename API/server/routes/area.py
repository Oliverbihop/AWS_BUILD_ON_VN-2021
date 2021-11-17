from fastapi import APIRouter, Body
from fastapi.encoders import jsonable_encoder
from server.models.reponse import Response


from server.database import (
    add_area,
    delete_area,
    retrieve_area,
    retrieve_areas,
    update_area,
)
from server.models.area import (
    AreaSchema,
    UpdateAreaSchema,
)


router = APIRouter()


@router.get("/{id}", response_description="Area data retrieved by ID")
async def get_student_data(id):
    area = await retrieve_area(id)
    if area:
        return Response.ResponseModel(area, "Area data retrieved successfully")
    return Response.ErrorResponseModel("An error occurred.", 404, "Area doesn't exist.")

@router.post("/", response_description="Add area into database")
async def add_student_data(area: AreaSchema = Body(...)):
    student = jsonable_encoder(area)
    new_area = await add_area(student)
    return Response.ResponseModel(new_area, "Area added successfully.")

@router.get("/", response_description="Areas retrieved")
async def get_students():
    areas = await retrieve_areas()
    if areas:
        return Response.ResponseModel(areas, "Area data retrieved successfully")
    return Response.ResponseModel(areas, "Empty list returned")


@router.put("/{id}")
async def update_area_data(id: str, req: UpdateAreaSchema = Body(...)):
    req = {k: v for k, v in req.dict().items() if v is not None}
    updated_area = await update_area(id, req)
    if updated_area:
        return Response.ResponseModel(
            "Area with ID: {}  update is successful".format(id),
            "Area updated successfully",
        )
    return Response.ErrorResponseModel(
        "An error occurred",
        404,
        "There was an error updating the area data.",
    )


@router.delete("/{id}", response_description="Delete Area from the database")
async def delete_area_data(id: str):
    deleted_area = await delete_area(id)
    if deleted_area:
        return Response.ResponseModel(
            "Area with ID: {} removed".format(id), "Area deleted successfully"
        )
    return Response.ErrorResponseModel(
        "An error occurred", 404, "Area with id {0} doesn't exist".format(id)
    )

