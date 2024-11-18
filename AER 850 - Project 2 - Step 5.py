# Aadhith Ravi
# 501045029
# AER 850 Project 2

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import  load_img, img_to_array

model = load_model('AadhithProject2.h5')

test_images = {
    "test_crack": "/Users/aadhi/Documents/GitHub/aadhith/Project 2 Data/Data/test/crack/test_crack.jpg",
    "test_missinghead": "/Users/aadhi/Documents/GitHub/aadhith/Project 2 Data/Data/test/missinghead/test_missinghead.jpg",
    "test_paintoff": "/Users/aadhi/Documents/GitHub/aadhith/Project 2 Data/Data/test/paintoff/test_paintoff.jpg"
}

class_labels = ["Crack", "Missing Screw Head", "Surface Degradation (Paint Off)"]

import numpy as np

def predict_image(image_path):
    img = load_img(image_path, target_size=(100, 100))
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) 
    img_array /= 255.0  

    predictions = model.predict(img_array)
    predicted_class = class_labels[np.argmax(predictions)]
    confidence = np.max(predictions)

    return predicted_class, confidence

import matplotlib.pyplot as plt


plt.figure(figsize=(16, 4))
for x, (label, image_path) in enumerate(test_images.items()):
    predicted_class, confidence = predict_image(image_path)
    
    img = load_img(image_path, target_size=(100, 100))
    
    plt.subplot(1, 3, x + 1)
    plt.imshow(img)
    plt.title(f"{predicted_class} ({confidence:.2f})")
    plt.axis('off')

plt.tight_layout()
plt.show()
