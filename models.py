from pydantic import BaseModel
from typing import Optional

class PredictRequest(BaseModel):
    output_image_name: Optional[str] = "output_image.png"
