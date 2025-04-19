# NSFW Image Detector

This project is an NSFW (Not Safe For Work) image detector using a pre-trained Vision Transformer model. It provides both a FastAPI backend and a Streamlit frontend for classifying images as NSFW or SFW (Safe For Work).

## Requirements

The project dependencies are listed in the `requirements.txt` file:

- fastapi
- uvicorn
- transformers
- pillow
- pydantic
- torch
- python-multipart
- streamlit

You can install the dependencies using the following command:

```sh
pip install -r [requirements.txt](http://_vscodecontentref_/0)
```
FastAPI Backend
The FastAPI backend is implemented in the nfsw.py file. It provides an endpoint to classify images.

Running the FastAPI Server
To run the FastAPI server, use the following command:

python [nfsw.py](http://_vscodecontentref_/1)

The server will start on http://0.0.0.0:8000.

API Endpoint
POST /classify: Classifies an uploaded image as NSFW or SFW.
Request
file: The image file to be classified.
Response
label: The classification label (NSFW or SFW).
score: The confidence score of the classification.

Streamlit Frontend
The Streamlit frontend is implemented in the streamlit_app.py file. It provides a web interface for uploading and classifying images.

Running the Streamlit App
To run the Streamlit app, use the following command:

```
streamlit run streamlit_app.py
```

The app will start on http://localhost:8501.

