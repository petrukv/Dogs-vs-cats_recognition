import tensorflow as tf
from keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt

model = tf.keras.models.load_model("dogs-vs-cats.h5")

img_path = "F:\Python mini projects\Coursera\dogs-vs-cats\dog.jpg"

img = image.load_img(img_path, target_size=(100, 100))
img_array = image.img_to_array(img)
img_array = img_array / 255.0  
img_array = np.expand_dims(img_array, axis=0) 


predictions = model.predict(img_array)
predicted_class_index = int(predictions[0][0] + 0.5) 
confidence = predictions[0][0] if predicted_class_index == 1 else 1 - predictions[0][0]


class_names = ["кіт", "собака"]


print(f"Передбачений клас: {class_names[predicted_class_index]}")
print(f"Впевненість: {confidence:.2%}")


plt.imshow(img)
plt.title(f"Передбачений клас: {class_names[predicted_class_index]}\nВпевненість: {confidence:.2%}")
plt.axis('off')
plt.show()
