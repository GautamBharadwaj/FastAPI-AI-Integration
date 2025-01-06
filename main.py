from fastapi import FastAPI, File, UploadFile, Query
from fastapi.responses import StreamingResponse
from utils import load_model, process_image, save_image, get_inference_results
from pydantic import BaseModel
from typing import Optional

class PredictRequest(BaseModel):
    output_image_name: Optional[str] = "output_image.png"

model = load_model("yolo11n.pt")
app = FastAPI()


@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    request: PredictRequest = Query(...),
):
    image_bytes = await file.read()
    image = process_image(image_bytes)
    output_image = get_inference_results(model, image)
    img_byte_arr = save_image(output_image, request.output_image_name)
    return StreamingResponse(img_byte_arr, media_type="image/png")
