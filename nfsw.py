from fastapi import FastAPI, UploadFile, File, HTTPException
from transformers import pipeline
from PIL import Image
import io
from pydantic import BaseModel
from PIL import UnidentifiedImageError


from fastapi.middleware.cors import CORSMiddleware

# Initialize the FastAPI app
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # This allows all origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# Load the pipeline once at startup
print("Loading pipeline...")
pipe = pipeline("image-classification", model="AdamCodd/vit-base-nsfw-detector")
print("Pipeline loaded successfully.")

class ClassificationResponse(BaseModel):
    label: str
    score: float

@app.post("/classify", response_model=ClassificationResponse)
async def classify_image(file: UploadFile = File(...)):
    try:
        # Read the file content
        contents = await file.read()
        # Create a file-like object from the bytes
        image_file = io.BytesIO(contents)
        # Open the image from the file-like object
        image = Image.open(image_file)
        # Make prediction
        result = pipe(image)
        # Get the top prediction
        top_result = max(result, key=lambda x: x['score'])
        # Return the top label and score
        return ClassificationResponse(label=top_result['label'], score=top_result['score'])
    except UnidentifiedImageError:
        raise HTTPException(status_code=400, detail="Invalid image file")
    except Exception as e:
        raise HTTPException(status_code=500, detail="Internal server error")

if __name__ == "__main__":
    import uvicorn
    import os
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)