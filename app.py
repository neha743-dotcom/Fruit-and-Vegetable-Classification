import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os
from datetime import datetime

st.set_page_config(
    page_title="Fruit & Vegetable Classifier",
    page_icon="🍎",
    layout="wide"
)

MODEL_PATH = "fruit_vegetable_model.keras"
TRAIN_DIR = "dataset/Training/Training"

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)

model = load_model()

class_names = sorted([
    folder
    for folder in os.listdir(TRAIN_DIR)
    if os.path.isdir(os.path.join(TRAIN_DIR, folder))
])

if "history" not in st.session_state:
    st.session_state.history = []

st.title("🍎 Fruit & Vegetable Classification System")

st.write(
    "CNN based Fruit and Vegetable Classification "
    "using the Fruits-360 dataset."
)

st.divider()

# SIDEBAR
with st.sidebar:
    st.header("Model Information")
    st.write("Model: CNN")
    st.write("Dataset: Fruits-360")
    st.write("Classes:", len(class_names))
    st.write("Image Size: 100 x 100")
    st.write("Epochs: 5")

# TABS
tab1, tab2, tab3, tab4 = st.tabs([
    "Single Prediction",
    "Multiple Images",
    "Prediction History",
    "Model Performance"
])

# =========================================================
# SINGLE PREDICTION
# =========================================================

with tab1:

    st.subheader("📷 Upload an Image")

    uploaded_file = st.file_uploader(
        "Choose a fruit or vegetable image",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:

        image = Image.open(uploaded_file).convert("RGB")

        st.image(
            image,
            caption="Uploaded Image",
            width=350
        )

        if st.button("🔍 Predict"):

            resized_image = image.resize((100, 100))

            image_array = np.array(resized_image)

            image_array = np.expand_dims(
                image_array,
                axis=0
            )

            predictions = model.predict(
                image_array,
                verbose=0
            )

            predicted_index = np.argmax(
                predictions[0]
            )

            predicted_class = class_names[
                predicted_index
            ]

            confidence = (
                predictions[0][predicted_index]
                * 100
            )

            st.success(
                "Prediction: " + predicted_class
            )

            st.info(
                "Confidence: "
                + str(round(confidence, 2))
                + "%"
            )

            st.progress(
                float(confidence / 100)
            )

            # TOP 5
            st.subheader("🏆 Top 5 Predictions")

            top_indices = np.argsort(
                predictions[0]
            )[-5:][::-1]

            for rank, index in enumerate(
                top_indices,
                start=1
            ):

                percentage = (
                    predictions[0][index]
                    * 100
                )

                st.write(
                    str(rank)
                    + ". "
                    + class_names[index]
                    + " - "
                    + str(round(percentage, 2))
                    + "%"
                )

            # HISTORY
            st.session_state.history.append({
                "time": datetime.now().strftime(
                    "%Y-%m-%d %H:%M:%S"
                ),
                "prediction": predicted_class,
                "confidence": round(
                    confidence,
                    2
                )
            })


# =========================================================
# MULTIPLE IMAGES
# =========================================================

with tab2:

    st.subheader("🖼️ Multiple Image Prediction")

    files = st.file_uploader(
        "Select multiple images",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True
    )

    if files:

        st.write(
            "Images selected:",
            len(files)
        )

        if st.button("🔍 Predict All Images"):

            for file in files:

                image = Image.open(
                    file
                ).convert("RGB")

                resized_image = image.resize(
                    (100, 100)
                )

                image_array = np.array(
                    resized_image
                )

                image_array = np.expand_dims(
                    image_array,
                    axis=0
                )

                predictions = model.predict(
                    image_array,
                    verbose=0
                )

                predicted_index = np.argmax(
                    predictions[0]
                )

                predicted_class = class_names[
                    predicted_index
                ]

                confidence = (
                    predictions[0][predicted_index]
                    * 100
                )

                st.image(
                    image,
                    width=200
                )

                st.success(
                    "Prediction: "
                    + predicted_class
                )

                st.write(
                    "Confidence: "
                    + str(round(confidence, 2))
                    + "%"
                )

                st.divider()


# =========================================================
# PREDICTION HISTORY
# =========================================================

with tab3:

    st.subheader("📋 Prediction History")

    if len(st.session_state.history) == 0:

        st.info(
            "No predictions available."
        )

    else:

        for item in reversed(
            st.session_state.history
        ):

            st.write(
                "Time: "
                + item["time"]
            )

            st.write(
                "Prediction: "
                + item["prediction"]
            )

            st.write(
                "Confidence: "
                + str(
                    item["confidence"]
                )
                + "%"
            )

            st.divider()

        if st.button("🗑️ Clear History"):

            st.session_state.history = []

            st.rerun()


# =========================================================
# MODEL PERFORMANCE
# =========================================================

with tab4:

    st.subheader(
        "📈 Model Performance"
    )

    st.write(
        "The following graphs show the "
        "training performance of the CNN model."
    )

    # ACCURACY GRAPH
    st.markdown(
        "### 📊 Training and Validation Accuracy"
    )

    if os.path.exists(
        "accuracy_graph.png"
    ):

        st.image(
            "accuracy_graph.png",
            caption="Training and Validation Accuracy",
            use_container_width=True
        )

    else:

        st.warning(
            "accuracy_graph.png not found."
        )

    st.divider()

    # LOSS GRAPH
    st.markdown(
        "### 📉 Training and Validation Loss"
    )

    if os.path.exists(
        "loss_graph.png"
    ):

        st.image(
            "loss_graph.png",
            caption="Training and Validation Loss",
            use_container_width=True
        )

    else:

        st.warning(
            "loss_graph.png not found."
        )


# =========================================================
# FOOTER
# =========================================================

st.divider()

st.caption(
    "Fruit & Vegetable Classification System | "
    "CNN + Fruits-360 Dataset"
)