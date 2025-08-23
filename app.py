import os
from flask import Flask, request, render_template
import tensorflow as tf
import numpy as np
from PIL import Image, ImageFile
from tensorflow.keras.layers import Dense, Conv2D, Activation, Add, Multiply, Concatenate, GlobalAveragePooling2D, Dropout
from tensorflow.keras.applications import ResNet101
from tensorflow.keras.applications.resnet import preprocess_input
from werkzeug.utils import secure_filename

ImageFile.LOAD_TRUNCATED_IMAGES = True

# ===== Define CBAM =====
@tf.keras.utils.register_keras_serializable(package="Custom", name="CBAM")
class CBAM(tf.keras.layers.Layer):
    def __init__(self, ratio=8, kernel_size=7, **kwargs):
        super(CBAM, self).__init__(**kwargs)
        self.ratio = ratio
        self.kernel_size = kernel_size

    def build(self, input_shape):
        channel = int(input_shape[-1])
        self.shared_dense_1 = Dense(channel // self.ratio, activation='relu', kernel_initializer='he_normal')
        self.shared_dense_2 = Dense(channel, kernel_initializer='he_normal')
        self.conv_spatial = Conv2D(filters=1, kernel_size=self.kernel_size, strides=1, padding='same',
                                   activation='sigmoid', kernel_initializer='he_normal', use_bias=False)
        super(CBAM, self).build(input_shape)

    def call(self, inputs):
        avg_pool = tf.reduce_mean(inputs, axis=[1, 2], keepdims=True)
        max_pool = tf.reduce_max(inputs, axis=[1, 2], keepdims=True)
        avg_out = self.shared_dense_2(self.shared_dense_1(avg_pool))
        max_out = self.shared_dense_2(self.shared_dense_1(max_pool))
        channel_attention = Activation('sigmoid')(Add()([avg_out, max_out]))
        channel_refined = Multiply()([inputs, channel_attention])

        avg_pool_spatial = tf.reduce_mean(channel_refined, axis=-1, keepdims=True)
        max_pool_spatial = tf.reduce_max(channel_refined, axis=-1, keepdims=True)
        concat = Concatenate(axis=-1)([avg_pool_spatial, max_pool_spatial])
        spatial_attention = self.conv_spatial(concat)

        refined_feature = Multiply()([channel_refined, spatial_attention])
        return refined_feature

    def get_config(self):
        config = super(CBAM, self).get_config()
        config.update({"ratio": self.ratio, "kernel_size": self.kernel_size})
        return config

# ===== Build the architecture =====
def build_model(input_shape=(224, 224, 3), num_classes=2):
    base_model = ResNet101(weights=None, include_top=False, input_shape=input_shape)
    x = base_model.output
    x = CBAM()(x)
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.5)(x)
    x = Dense(256, activation='relu')(x)
    output = Dense(num_classes, activation='softmax')(x)
    model = tf.keras.models.Model(inputs=base_model.input, outputs=output)
    return model

# ===== Load the weights from the project directory =====
MODEL_PATH = os.path.join(os.path.dirname(__file__), "model", "fraction-model.keras")

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"❌ Model file not found: {MODEL_PATH}")

model = build_model()
model.load_weights(MODEL_PATH)

print("✅ Model architecture rebuilt and weights loaded successfully.")

class_names = ["fractured", "not fractured"]

def prepare_image(image_path, target_size=(224, 224)):
    img = Image.open(image_path).convert("RGB")
    img = img.resize(target_size)
    img_array = np.expand_dims(np.array(img), axis=0)
    img_array = preprocess_input(img_array)
    return img_array

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def upload_predict():
    if request.method == "POST":
        if "file" not in request.files:
            return "⚠️ No file uploaded", 400
        file = request.files["file"]
        if file.filename == "":
            return "⚠️ Empty filename", 400
        if file:
            upload_folder = os.path.join(app.root_path, 'static', 'uploads')
            os.makedirs(upload_folder, exist_ok=True)

            filename = secure_filename(file.filename)
            filepath = os.path.join(upload_folder, filename)
            file.save(filepath)

            img_array = prepare_image(filepath)
            preds = model.predict(img_array)
            pred_class = class_names[np.argmax(preds)]
            confidence = float(np.max(preds))

            return render_template("index.html",
                                   prediction=pred_class,
                                   confidence=confidence,
                                   filename=filename)
    return render_template("index.html", prediction=None)

if __name__ == "__main__":
    app.run(debug=True)
