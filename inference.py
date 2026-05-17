import tensorflow as tf
import numpy as np
from PIL import Image
import pickle


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


model = tf.keras.models.load_model(
    "best_model.keras",
    custom_objects={
        "RandomBackground": RandomBackground
    },
    compile=False
)


with open(
    "class_names.pkl",
    "rb"
) as f:

    class_names = pickle.load(f)


def preprocess_image(image_path):

    image = Image.open(
        image_path
    ).convert("RGB")

    image = image.resize(
        (224,224)
    )

    img = np.array(image)

    img = tf.keras.applications.efficientnet.preprocess_input(
        img
    )

    img = np.expand_dims(
        img,
        axis=0
    )

    return img


def predict_image(
    image_path,
    top_k=3
):

    img = preprocess_image(
        image_path
    )

    pred = model.predict(
        img,
        verbose=0
    )[0]

    top_indices = pred.argsort()[-top_k:][::-1]

    print("\nTOP PREDICTIONS\n")

    for i in top_indices:

        label = class_names[i]

        confidence = pred[i]

        print(
            f"{label} : {confidence:.4f}"
        )


predict_image(
    "penyakit jagung.jpg"
)