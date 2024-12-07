import numpy as np
from keras.models import load_model
import streamlit as st
from PIL import Image

model = load_model('snakespecies.h5')

CLASS_NAMES = ['Common Krait',
                'King Cobra',
                'Monocled Cobra',
                "Russell's Viper",
                'Saw-scaled Viper',
                'Spectacled Cobra',
                'Banded Racer',
                'Checkered Keelback',
                'Common Rat Snake',
                'Common Sand Boa',
                'Common Trinket',
                'Green Tree Vine',
                'Indian Rock Python']

venomous = ['Common Krait', 'King Cobra', 'Monocled Cobra', "Russell's Viper", 'Saw-scaled Viper', 'Spectacled Cobra']
non_venomous = ['Banded Racer', 'Checkered Keelback', 'Common Rat Snake', 'Common Sand Boa', 'Common Trinket', 'Green Tree Vine', 'Indian Rock Python']

kind = ""
 #Setting Title of App
st.title("Snake Species Identification 🐍")
st.markdown("Upload an image of the Snake")

# Uploading the plant image
plant_image = st.file_uploader("Choose an image...", type="jpg")
submit = st.button('Predict')

# On predict button click
if submit:
    if plant_image is not None:
        # Open image using Pillow
        image = Image.open(plant_image)

        # Displaying the image
        st.image(image, caption='Uploaded Image', use_container_width=True)

        # Resizing the image
        image = image.resize((299, 299))

        # Convert image to numpy array
        image_np = np.array(image)

        # Expand dimensions to match model input shape
        image_np = np.expand_dims(image_np, axis=0)

        # Make prediction
        Y_pred = model.predict(image_np)

        # Display the prediction
        predicted_class = CLASS_NAMES[np.argmax(Y_pred)]
        if predicted_class in venomous:
            kind = "Venomous"
        else:
            kind = "Non-Venomous"
        st.title(f"The Snake Species is {predicted_class} it is {kind}")

# Footer
st.markdown("""
    <style>
        .footer {
            position: fixed;
            bottom: 0;
            left: 100;
            width: 100%;
            text-align: left;
            padding: 10px 0;
        }
    </style>
    <div class="footer">
        Made by Ajay Singh Rathore
    </div>
""", unsafe_allow_html=True)
