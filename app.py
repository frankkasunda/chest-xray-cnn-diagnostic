import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np

# 1. Setup the Page Design
st.set_page_config(page_title="Pneumonia Diagnostic Tool", page_icon="🫁")
st.title("Pediatric Pneumonia Diagnostic Assistant 🫁")
st.write("Upload a pediatric anterior-posterior chest X-ray to receive an automated AI screening.")

# 2. Cache the model so it doesn't reload every time a user uploads an image
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model('pneumonia_detection_model.h5')
    return model

model = load_model()

# 3. Create the File Uploader
uploaded_file = st.file_uploader("Choose an X-ray image (JPG/PNG)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='Uploaded Chest X-Ray', use_column_width=True)
    
    st.write("Analyzing scan...")
    
    # 4. Preprocess the image to match the Colab training pipeline exactly
    # Resize to 224x224
    size = (224, 224)
    image_resized = ImageOps.fit(image, size, Image.Resampling.LANCZOS)
    
    # Convert to numpy array and normalize pixel values (0-1)
    image_array = np.asarray(image_resized)
    normalized_image_array = image_array.astype(np.float32) / 255.0
    
    # Expand dimensions so the model thinks it's a "batch" of 1 image
    input_data = np.expand_dims(normalized_image_array, axis=0)
    
    # 5. Run the Prediction
    prediction = model.predict(input_data)
    probability = prediction[0][0]
    
    # 6. Display the Clinical Output
    st.markdown("---")
    if probability > 0.5:
        confidence = probability * 100
        st.error(f"**Diagnosis:** Pneumonia Detected")
        st.write(f"**AI Confidence Score:** {confidence:.2f}%")
        st.info("Recommendation: Please review scan for radiopaque consolidations and consult a pediatric pulmonologist.")
    else:
        # Since 1 is Pneumonia, the probability of Normal is (1 - probability)
        confidence = (1 - probability) * 100
        st.success(f"**Diagnosis:** Normal (No Pneumonia Detected)")
        st.write(f"**AI Confidence Score:** {confidence:.2f}%")
        st.info("Recommendation: Standard clinical follow-up.")
