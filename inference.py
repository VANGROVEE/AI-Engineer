import os
import random
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
    image_path=None,
    image_folder="test_image",
    top_k=3,
    num_images=3
):

    if image_path is None:
        images = [
            os.path.join(image_folder, f)
            for f in os.listdir(image_folder)
            if os.path.isfile(os.path.join(image_folder, f))
        ]
        if not images:
            raise FileNotFoundError(
                f"No image files found in folder: {image_folder}"
            )
        if num_images > len(images):
            raise ValueError(
                f"Requested {num_images} images but only {len(images)} available in {image_folder}"
            )
        images = random.sample(images, num_images)
    else:
        images = [image_path]

    for image_path in images:
        print(f"\nSelected image: {image_path}")

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


predict_image()