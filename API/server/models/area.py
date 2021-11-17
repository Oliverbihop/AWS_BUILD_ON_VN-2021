from typing import Optional

from pydantic import BaseModel, Field


class AreaSchema(BaseModel):
    """Area model."""
    name: str = Field(..., title="Name", description="Area name")
    description: Optional[str] = Field(None, title="Description", description="Area description")
    current_images: Optional[str] = Field(None, title="Current Images", description="Current Area images")
    previous_images: Optional[str] = Field(None, title="Previous Images", description="Previous Area images")
    current_objects: Optional[dict] = Field(None, title="Current Objects", description=" Current Area objects")
    previous_objects: Optional[dict] = Field(None, title="Previous Objects", description="Previous Area objects")

    class Config:
        schema_extra = {
            "example" : {
                "name" : "A1",
                "description" : "Area A1",
                "current_images" : "A",
                "previous_images" : "",
                "current_objects" : {},
                "previous_objects" : {}
            }
        }

class UpdateAreaSchema(BaseModel):
    """Area model."""
    name: str = Field(..., title="Name", description="Area name")
    description: Optional[str] = Field(None, title="Description", description="Area description")
    current_images: Optional[str] = Field(None, title="Current Images", description="Current Area images")
    previous_images: Optional[str] = Field(None, title="Previous Images", description="Previous Area images")
    current_objects: Optional[dict] = Field(None, title="Current Objects", description=" Current Area objects")
    previous_objects: Optional[dict] = Field(None, title="Previous Objects", description="Previous Area objects")

    class Config:
        schema_extra = {
            "example" : {
                "name" : "A1",
                "description" : "Area A1",
                "current_images" : "A",
                "previous_images" : "",
                "current_objects" : {},
                "previous_objects" : {}
            }
        }

