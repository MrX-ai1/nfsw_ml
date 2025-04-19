import streamlit as st
from transformers import pipeline
from PIL import Image, UnidentifiedImageError

@st.cache(allow_output_mutation=True)
def load_model():
    return pipeline("image-classification", model="AdamCodd/vit-base-nsfw-detector")

pipe = load_model()

# Sidebar
st.sidebar.title("NSFW Image Detector")
st.sidebar.write("This app uses a pre-trained Vision Transformer model to classify images as NSFW (Not Safe For Work) or SFW (Safe For Work).")

# Main content
st.title("NSFW Image Detector")
st.write("Upload an image to classify it as NSFW or SFW.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    st.image(uploaded_file, caption='Uploaded Image', use_column_width=True)
    with st.spinner('Classifying...'):
        try:
            image = Image.open(uploaded_file)
            result = pipe(image)
            top_result = max(result, key=lambda x: x['score'])
            if top_result['label'] == 'NSFW':
                st.error(f"Classification: {top_result['label']}")
            else:
                st.success(f"Classification: {top_result['label']}")
            st.write(f"Confidence: {top_result['score']:.2f}")
        except UnidentifiedImageError:
            st.error("Invalid image file. Please upload a valid image.")
        except Exception as e:
            st.error("An error occurred while processing the image.")
else:
    st.info("Please upload an image file to get started.")