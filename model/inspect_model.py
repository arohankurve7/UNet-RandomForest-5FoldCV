import tensorflow as tf
from tensorflow.keras.utils import plot_model

# load trained model
model = tf.keras.models.load_model("dental_xray_seg.h5")

# print architecture summary
model.summary()

# generate architecture diagram
plot_model(
    model,
    to_file="unet_architecture.png",
    show_shapes=True,
    show_layer_names=True
)

print("Architecture diagram saved as unet_architecture.png")