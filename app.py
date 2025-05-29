import streamlit as st
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image

# Load model
model = load_model("skincare.h5")

# Label klasifikasi
labels = ['Dry', 'Normal', 'Oily']

# Rekomendasi skincare
skincare_recommendations = {
    'Dry': {
        'skincare': [
            "Hydrating Cleanser",
            "Hyaluronic Acid Serum",
            "Rich Moisturizer",
            "Gentle Exfoliator (1x/minggu)"
        ],
        'tips': [
            "Gunakan pelembap setelah cuci muka",
            "Hindari sabun wajah yang membuat kulit kering",
            "Gunakan humidifier di ruangan ber-AC"
        ]
    },
    'Normal': {
        'skincare': [
            "Gentle Cleanser",
            "Light Moisturizer",
            "Sunscreen SPF 30+",
            "Weekly Exfoliation"
        ],
        'tips': [
            "Jaga pola makan dan tidur",
            "Rutin skincare pagi dan malam",
            "Pilih produk yang ringan dan tidak mengiritasi"
        ]
    },
    'Oily': {
        'skincare': [
            "Foaming Cleanser",
            "Salicylic Acid Toner",
            "Oil-Free Moisturizer",
            "Clay Mask (2x/minggu)"
        ],
        'tips': [
            "Cuci muka maksimal 2x sehari",
            "Gunakan kertas minyak saat perlu",
            "Pilih produk non-komedogenik"
        ]
    }
}

def is_face_detected(image):
    img = np.array(image.convert('RGB'))
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
    return len(faces) > 0

def classify_skin(image):
    if not is_face_detected(image):
        return "❌ Gambar tidak mengandung wajah. Harap upload ulang dengan wajah yang terlihat jelas."

    img = image.resize((224, 224))
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)[0]
    index = np.argmax(prediction)
    skin_type = labels[index]
    rec = skincare_recommendations[skin_type]

    return skin_type, rec['skincare'], rec['tips']

st.title("Deteksi Jenis Kulit Wajah")
st.write("Upload foto wajah kamu untuk mengetahui jenis kulit dan mendapatkan rekomendasi skincare!")

uploaded_file = st.file_uploader("Upload gambar wajah", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Gambar yang diupload', use_column_width=True)

    with st.spinner("Mendeteksi jenis kulit..."):
        result = classify_skin(image)
        if isinstance(result, str):
            st.error(result)
        else:
            skin_type, skincare_list, tips_list = result
            st.success(f"Jenis kulit terdeteksi: **{skin_type}**")
            st.markdown("### 🧴 Rekomendasi Skincare:")
            for item in skincare_list:
                st.write("- " + item)
            st.markdown("### 💡 Tips Perawatan:")
            for tip in tips_list:
                st.write("- " + tip)
