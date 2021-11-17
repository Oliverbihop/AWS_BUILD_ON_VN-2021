from typing import Optional

from pydantic import BaseModel, Field




class UnetSchema(BaseModel):
    """Unet Predict model."""

    hashImage: str = Field(..., title="Hash Image", description="Hash Image")

    class Config:
        schema_extra = {
            "example" : {
                "hashImage" : "A1",
            }
        }
