import gradio as gr
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

# Fungsi deteksi wajah dengan OpenCV
def is_face_detected(image):
    # Convert to OpenCV format
    img = np.array(image.convert('RGB'))
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # Load Haar Cascade detector
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    return len(faces) > 0

# Fungsi klasifikasi kulit
def classify_skin(image):
    if not is_face_detected(image):
        return "❌ **Gambar tidak mengandung wajah. Harap upload ulang dengan wajah yang terlihat jelas.**"

    # Preprocess image
    img = image.resize((224, 224))
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    prediction = model.predict(img_array)[0]
    index = np.argmax(prediction)
    skin_type = labels[index]
    recommendation = skincare_recommendations[skin_type]

    skincare_list = "\n- ".join(recommendation['skincare'])
    tips_list = "\n- ".join(recommendation['tips'])

    # Format output dengan markdown agar tampil beda dan rapi
    output = (
        f"🧑 **Jenis Kulit Terdeteksi:** {skin_type}\n\n"
        f"🧴 **Rekomendasi Skincare:**\n- {skincare_list}\n\n"
        f"💡 **Tips Perawatan:**\n- {tips_list}"
    )
    return output

# Gradio Interface
demo = gr.Interface(
    fn=classify_skin,
    inputs=gr.Image(type="pil"),
    outputs="markdown",
    title="Deteksi Jenis Kulit Wajah",
    description="Upload foto wajah untuk mengetahui jenis kulit (Dry, Normal, Oily) dan mendapatkan rekomendasi skincare."
)

if __name__ == "__main__":
    demo.launch()
