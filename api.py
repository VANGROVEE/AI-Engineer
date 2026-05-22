import io
import os
import pickle
from typing import List

import numpy as np
import tensorflow as tf
import google.generativeai as genai

from fastapi import (
    FastAPI,
    HTTPException
)

from fastapi.responses import JSONResponse
import requests
from PIL import Image
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()

GEMINI_API_KEY = os.getenv(
    "GEMINI_API_KEY"
)

genai.configure(
    api_key=GEMINI_API_KEY
)

gemini_model = genai.GenerativeModel(
    "gemini-3.1-flash-lite"
)


@tf.keras.utils.register_keras_serializable()
class RandomBackground(
    tf.keras.layers.Layer
):

    def call(self, images):

        images = tf.cast(
            images,
            tf.float32
        )

        gray = tf.reduce_mean(
            images,
            axis=-1,
            keepdims=True
        )

        mask = tf.cast(
            gray > 240,
            tf.float32
        )

        random_bg = tf.random.uniform(
            tf.shape(images),
            0,
            255
        )

        images = (
            images * (1-mask)
            +
            random_bg * mask
        )

        return images


def load_class_names(
    pickle_path: str
) -> List[str]:

    with open(
        pickle_path,
        "rb"
    ) as f:

        return pickle.load(f)

class PredictRequest(
    BaseModel
):

    image_url: str

    top_k: int = 3

    explain: bool = True

def preprocess_image_bytes(
    image_bytes: bytes
) -> np.ndarray:

    image = Image.open(
        io.BytesIO(image_bytes)
    ).convert("RGB")

    image = image.resize(
        (224,224)
    )

    image_array = np.array(
        image
    )

    image_array = tf.keras.applications.efficientnet.preprocess_input(
        image_array
    )

    return np.expand_dims(
        image_array,
        axis=0
    )


def build_prompt(
    label: str
):

    return f"""
Penyakit tanaman terdeteksi:
{label}

Berikan penjelasan dalam Bahasa Indonesia dengan format berikut:

Penyakit:
Penjelasan singkat mengenai penyakit.

Penyebab:
Penyebab utama penyakit.

Penanganan:
Langkah-langkah penanganan penyakit.

Pencegahan:
Cara mencegah penyakit.

Pemulihan:
Apakah tanaman masih bisa pulih atau tidak.

Gunakan bahasa yang singkat, jelas, rapi, dan mudah dipahami.

Jangan:
- memperkenalkan diri
- menyebut AI
- menyebut plant pathologist
- menggunakan markdown
- menggunakan paragraf panjang
"""


def generate_ai_response(
    label: str
):

    prompt = build_prompt(
        label
    )

    response = gemini_model.generate_content(
        prompt
    )

    cleaned = response.text.replace(
        "*",
        ""
    )

    return cleaned



app = FastAPI(
    title="Plant Disease AI API",
    version="1.0.0"
)


model = tf.keras.models.load_model(
    "best_model_finetune.keras",
    custom_objects={
        "RandomBackground": RandomBackground
    },
    compile=False
)


class_names = load_class_names(
    "class_names.pkl"
)

LABEL_MAPPING = {

    "corn_blight":
    "Corn Blight",

    "corn_healthy":
    "Healthy Corn",

    "corn_leaf_spot":
    "Corn Leaf Spot",

    "corn_rust":
    "Corn Rust",


    "mango_anthrecnose":
    "Mango Anthracnose",

    "mango_dieback":
    "Mango Dieback",

    "mango_gall_mildge_damage":
    "Mango Gall Mildge Damage",

    "mango_healthy":
    "Healthy Mango",

    "mango_insect_damage_webber":
    "Mango Insect Damage",

    "mango_leaf_blight":
    "Mango Leaf Blight",


    "potato_bacteria":
    "Potato Bacterial Disease",

    "potato_fungi":
    "Potato Fungal Disease",

    "potato_healthy":
    "Healthy Potato",

    "potato_pest":
    "Potato Pest Damage",

    "potato_phytophora":
    "Potato Phytophthora",

    "potato_virus":
    "Potato Virus",


    "tomato_bacterial_spot":
    "Tomato Bacterial Spot",

    "tomato_early_blight":
    "Tomato Early Blight",

    "tomato_healthy":
    "Healthy Tomato",

    "tomato_late_blight":
    "Tomato Late Blight",

    "tomato_leaf_mold":
    "Tomato Leaf Mold",

    "tomato_powdery_mildew":
    "Tomato Powdery Mildew",

    "tomato_septoria_leaf_spot":
    "Tomato Septoria Leaf Spot",

    "tomato_spider_mites_two_spotted_spider_mite":
    "Tomato Spider Mites",

    "tomato_target_spot":
    "Tomato Target Spot",

    "tomato_tomato_mosaic_virus":
    "Tomato Mosaic Virus",

    "tomato_tomato_yellow_leaf_curl_virus":
    "Tomato Yellow Leaf Curl Virus"

}

@app.get("/")
def root():

    return {
        "service": "Plant Disease AI API",
        "status": "running"
    }


def split_plant_disease(
    disease_name: str
):

    if "Corn" in disease_name:

        return "Jagung", disease_name

    elif "Mango" in disease_name:

        return "Mangga", disease_name

    elif "Potato" in disease_name:

        return "Kentang", disease_name

    elif "Tomato" in disease_name:

        return "Tomat", disease_name

    return "Unknown", disease_name


@app.post("/predict")
async def predict(
    request: PredictRequest
):

    try:

        response = requests.get(
            request.image_url
        )

        image_bytes = response.content

    except Exception as exc:

        raise HTTPException(
            status_code=400,
            detail=f"Failed to download image: {exc}"
        )

    try:

        input_tensor = preprocess_image_bytes(
            image_bytes
        )

    except Exception as exc:

        raise HTTPException(
            status_code=400,
            detail=f"Image preprocessing failed: {exc}"
        )

    predictions = model.predict(
        input_tensor,
        verbose=0
    )[0]

    top_k = max(
        1,
        min(
            request.top_k,
            len(predictions)
        )
    )

    top_indices = predictions.argsort()[
        -top_k:
    ][::-1]

    results = []

    for index in top_indices:

        disease_name = LABEL_MAPPING.get(
            class_names[int(index)],
            class_names[int(index)]
        )

        plant_name, clean_disease = split_plant_disease(
            disease_name
        )

        results.append({

            "nama_tanaman":
            plant_name,

            "nama_penyakit":
            clean_disease,

            "confidence":
            float(predictions[int(index)])

        })

    top_result = results[0]

    top_plant = top_result[
        "nama_tanaman"
    ]

    top_disease = top_result[
        "nama_penyakit"
    ]

    top_confidence = top_result[
        "confidence"
    ]

    if top_confidence < 0.5:

        response_json = {

            "nama_tanaman":
            "Unknown",

            "nama_penyakit":
            "Unknown",

            "confidence":
            top_confidence,

            "message":
            (
                "Model tidak cukup yakin untuk "
                "mengidentifikasi penyakit. "
                "Kemungkinan gambar kurang jelas "
                "atau penyakit tidak tersedia "
                "di dataset training."
            )

        }

        if request.explain:

            response_json[
                "ai_explanation"
            ] = (
                "Silakan upload gambar daun "
                "yang lebih jelas dengan pencahayaan "
                "yang baik. Bisa juga penyakit "
                "tersebut belum tersedia di "
                "dataset AI saat ini."
            )

        return JSONResponse(
            response_json
        )

    response_json = {

        "nama_tanaman":
        top_plant,

        "nama_penyakit":
        top_disease,

        "confidence":
        top_confidence,

        "predictions":
        results

    }

    if request.explain:

        try:

            ai_response = generate_ai_response(
                top_disease
            )

            response_json[
                "ai_explanation"
            ] = ai_response

        except Exception as exc:

            response_json[
                "ai_error"
            ] = str(exc)

    return JSONResponse(
        response_json
    )


if __name__ == "__main__":

    import uvicorn

    uvicorn.run(
        "api:app",
        host="127.0.0.1",
        port=8000,
        reload=True
    )