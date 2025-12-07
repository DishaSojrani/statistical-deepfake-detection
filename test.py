import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt

# Load the saved model
model = tf.keras.models.load_model('Blood1.keras')
x=450
def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(x, x))  # Adjust size to match model input
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array /= 255.0  # Normalize to [0, 1]
    return img_array

def predict_image(img_path):
    img_array = preprocess_image(img_path)
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions, axis=1)
    return predicted_class, predictions
# Specify the path to the image you want to predict
image_path = '444.jpg'
className=['Eosinophil','Lymphocyte','Basophils','Erythroblast']
#className=['Basophils','Eosinophil','Erythroblast','Lymphocyte']
# Make a prediction
predicted_class, predictions = predict_image(image_path)
print(predicted_class)
# Display the image
img = image.load_img(image_path)
plt.imshow(img)
plt.axis('off')
plt.title(f'Predicted Class: {className[predicted_class[0]]}')
plt.show()

# Show the prediction probabilities for each class
print('Prediction probabilities:', predictions)
