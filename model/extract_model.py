import tensorflow as tf

# load model
model = tf.keras.models.load_model("model/dental_xray_seg.h5")

# print architecture
model.summary()

# save architecture JSON
with open("model_architecture.json", "w") as f:
    f.write(model.to_json())

print("Architecture saved to model_architecture.json")