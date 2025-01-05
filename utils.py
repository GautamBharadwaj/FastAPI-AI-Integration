from io import BytesIO
from PIL import Image
from ultralytics import YOLO


def load_model(model_path: str) -> YOLO:
    """Load the YOLO model."""
    return YOLO(model_path)


def process_image(image_bytes: bytes) -> Image:
    """Convert byte stream to PIL image."""
    image = Image.open(BytesIO(image_bytes))
    return image


def save_image(image: Image, output_image_name: str) -> BytesIO:
    """Save the output image to a byte stream and also save it locally."""
    img_byte_arr = BytesIO()
    image.save(img_byte_arr, format="PNG")
    image.save(output_image_name)  # Save the image locally with the given name
    img_byte_arr.seek(0)
    return img_byte_arr


def get_inference_results(model: YOLO, image: Image) -> Image:
    """Run the YOLO model on the image and return the result as a PIL image."""
    results = model(image)
    output_array = results[0].plot()  # This gives a NumPy array
    output_image = Image.fromarray(output_array)
    return output_image
