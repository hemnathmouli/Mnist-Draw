import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

mnist = tf.keras.datasets.mnist

# Downloading the mnist Data
(x_train, y_train),(x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

# 4 layers from converting the data into Flat to giving out softmax 0-9 result
model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(),
  tf.keras.layers.Dense(512, activation=tf.nn.relu),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(10, activation=tf.nn.softmax)
])
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Training the model
model.fit(x_train, y_train, epochs=10)

score, acc = model.evaluate(x_test, y_test)

print("The score is %s" % (score))
print("The accuracy is %s" % (acc))

# Show first test data
plt.title('Testing value for')
plt.imshow(x_test[0], cmap='gray')
plt.show()

# Prediction the first test data
predicted = model.predict(x_test[0].reshape(1,784))
predicted = np.argmax(predicted)

print("The predicted test data value is: %d" % predicted)

from main import runall

cv = runall(model)
cv.compile()