python -m pip install --upgrade pip

import streamlit as st
from PIL import Image, ImageFilter
import requests
from io import BytesIO
import numpy as np
import cv2


@st.cache
def load_image(image_file):
    img = Image.open(image_file)
    return img


def image_to_array(image):
    img_array = np.array(image.convert("RGB"))
    img_array = img_array[:, :, ::-1].copy()
    return img_array


def array_to_image(img_array):
    img = Image.fromarray(np.uint8(img_array))
    return img


def morphing(father_image, mother_image, child_ratio=0.5):
    father_array = image_to_array(father_image)
    mother_array = image_to_array(mother_image)

    assert father_array.shape == mother_array.shape

    mask = np.zeros(father_array.shape, dtype=np.float32)
    mask[:, :int(father_array.shape[1]*child_ratio), :] = 1

    father_mask = father_array * mask
    mother_mask = mother_array * (1 - mask)

    child_array = father_mask + mother_mask
    child_image = array_to_image(child_array)

    return child_image


def main():
    st.title("Generate a Child's Photo")

    st.write("Please upload photos of father and mother.")

    father_image = st.file_uploader("Father's Photo", type=["jpg", "jpeg", "png"])
    mother_image = st.file_uploader("Mother's Photo", type=["jpg", "jpeg", "png"])

    if father_image and mother_image:
        col1, col2 = st.beta_columns(2)
        with col1:
            st.write("Father's Photo")
            st.image(father_image, use_column_width=True)

        with col2:
            st.write("Mother's Photo")
            st.image(mother_image, use_column_width=True)

        if st.button("Generate"):
            child_image = morphing(father_image, mother_image)
            st.image(child_image, use_column_width=True)


if __name__ == "__main__":
    main()
