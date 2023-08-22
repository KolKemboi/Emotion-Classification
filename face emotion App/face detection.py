import os
import cv2
import matplotlib.pyplot as plt
import tensorflow as tf
import keras
import numpy as np

classifier = keras.models.load_model("C:\\Users\\USER\\Desktop\\AI task\\website\\face_emotion\\face_emotion_nn")

CLASS_NAMES = "ANGRY,DISGUSTED,FEARFUL,HAPPY,NEUTRAL,SAD,SURPRISED".lower().split(",")

path = "img_1.jpg"

print(classifier)



image = cv2.imread(path, 0)

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
faces = face_cascade.detectMultiScale(image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

for (x, y, w, h) in faces:
	cv2.rectangle(image, (x, y), (x + w, y + h), (255, 255, 255), 3)
	frame = image[y:y + h, x:x + w]

image_path = os.path.join("to_classify", "image.jpg")

#image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

plt.imshow(image, cmap = "gray")
plt.show()

cv2.imwrite(image_path, frame)

img = tf.keras.utils.load_img(image_path,
    grayscale=True,
    color_mode='grayscale',
    target_size = (48, 48)
)
img_array = tf.keras.utils.img_to_array(img)
img_array = tf.expand_dims(img_array, 0)

prediction = classifier.predict(img_array)
score = tf.nn.softmax(prediction[0])
print("this image belongs to {}, with a {:.2f} percent accuracy"
      .format(CLASS_NAMES[np.argmax(score)], 100*np.max(score)))

