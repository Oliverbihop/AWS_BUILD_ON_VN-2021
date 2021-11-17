import motor.motor_asyncio
from bson.objectid import ObjectId


MONGO_DETAILS = "mongodb://localhost:27017"

client = motor.motor_asyncio.AsyncIOMotorClient(MONGO_DETAILS)

database = client.area

area_collection = database.get_collection("area_collection")


#Helper
def areaHelper(area) -> dict:
    return {
        "id": str(area["_id"]),
        "name": area["name"],
        "description": area["description"],
        "current_images" : area["current_images"],
        "previous_images" :area["previous_images"],
        "current_objects" : area["current_objects"],
        "previous_objects" : area["previous_objects"]
    }



# Retrieve all students present in the database
async def retrieve_areas():
    areas = []
    async for areas in area_collection.find():
        areas.append(areaHelper(areas))
    return areas


# Add a new area into to the database
async def add_area(area_data: dict) -> dict:
    area = await area_collection.insert_one(area_data)
    new_area = await area_collection.find_one({"_id": area.inserted_id})
    return areaHelper(new_area)

async def retrieve_area(id: str) -> dict:
    area = await area_collection.find_one({"_id": ObjectId(id)})
    if area:
        return areaHelper(area)

async def update_area(id: str, data: dict):
    # Return false if an empty request body is sent.
    if len(data) < 1:
        return False
    area = await area_collection.find_one({"_id": ObjectId(id)})
    if area:
        updated_area = await area_collection.update_one(
            {"_id": ObjectId(id)}, {"$set": data}
        )
        if updated_area:
            return True
        return False

# Delete a student from the database
async def delete_area(id: str):
    area = await area_collection.find_one({"_id": ObjectId(id)})
    if area:
        await area_collection.delete_one({"_id": ObjectId(id)})
        return True