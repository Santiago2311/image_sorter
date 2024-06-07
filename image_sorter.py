import math
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds

# Zalando's Fashion MNIST Data Set
data, metadata = tfds.load('fashion_mnist', as_supervised=True, with_info=True)

# Separation of training data (60k) and test data (10k)
train, test = data['train'], data['test']

# Labels of the 10 possible categories
name_class = metadata.features['label'].names

# Normalization function for the data (from 0-255 to 0-1)
# Makes the network learn better and faster


def normalize(images, labels):
    images = tf.cast(images, tf.float32)
    images /= 255
    return images, labels


train = train.map(normalize)
test = test.map(normalize)

# Add to cache (using memory instead of disk achieves faster training)
train = train.cache()
test = test.cache()

# Creation of the model
model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28, 1)),  # 1 - blanco y negro
    tf.keras.layers.Dense(50, activation=tf.nn.relu),
    tf.keras.layers.Dense(50, activation=tf.nn.relu),
    # Para redes de clasificacion
    tf.keras.layers.Dense(10, activation=tf.nn.softmax)
])

# Model compilation
model.compile(
    optimizer='adam',
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    metrics=['accuracy']
)

n_train = metadata.splits["train"].num_examples
n_test = metadata.splits["test"].num_examples

# Batch work allows training with large amounts of data to be done
# more efficiently.
LOT_SIZE = 32

# Shuffle and repeat cause the data to be randomly shuffled so that the network
# does not learn the order of things.
train = train.repeat().shuffle(n_train).batch(LOT_SIZE)
test = test.batch(LOT_SIZE)

history = model.fit(train, epochs=5,
                    steps_per_epoch=math.ceil(n_train/LOT_SIZE))

plt.xlabel("# Epochs")
plt.ylabel("Loss size")
plt.plot(history.history["loss"])

# Create a grid with several predictions, and mark whether
# it was correct (blue) or incorrect (red).

for test_image, test_label in test.take(1):
    test_image = test_image.numpy()
    test_label = test_label.numpy()
    prediction = model.predict(test_image)


def plot_image(i, arr_predictions, real_labels, images):
    arr_predictions, real_label, img = arr_predictions[i], real_labels[i], images[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])

    plt.imshow(img[..., 0], cmap=plt.cm.binary)

    predicted_label = np.argmax(arr_predictions)
    if predicted_label == real_label:
        color = 'blue'
    else:
        color = 'red'

    plt.xlabel("{} {:2.0f}% ({})".format(name_class[predicted_label],
                                         100*np.max(arr_predictions),
                                         name_class[real_label]),
               color=color)


def plot_array_value(i, arr_predicciones, etiqueta_real):
    arr_predicciones, etiqueta_real = arr_predicciones[i], etiqueta_real[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    grafica = plt.bar(range(10), arr_predicciones, color="#777777")
    plt.ylim([0, 1])
    etiqueta_prediccion = np.argmax(arr_predicciones)

    grafica[etiqueta_prediccion].set_color('red')
    grafica[etiqueta_real].set_color('blue')


row = 5
columns = 5
n_images = row*columns
plt.figure(figsize=(2*2*columns, 2*row))
for i in range(n_images):
    plt.subplot(row, 2*columns, 2*i+1)
    plot_image(i, prediction, test_label, test_image)
    plt.subplot(row, 2*columns, 2*i+2)
    plot_array_value(i, prediction, test_label)

# Prediction test
image = test_image[4]
image = np.array([image])
prediction = model.predict(image)

print("Prediction: " + name_class[np.argmax(prediction[0])])
