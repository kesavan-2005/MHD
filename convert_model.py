import tensorflow as tf

# Load your original CNN model (already saved in .keras format)
model = tf.keras.models.load_model(r"C:\Users\kesav\PycharmProjects\MHD\models\cnn_model.keras")

# Save the model again in HDF5 (.h5) format
model.save("cnn_model.h5", save_format="h5")

print("✅ Model successfully saved as cnn_model.h5")
