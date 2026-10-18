from pathlib import Path

import numpy as np
from PIL import Image
import time

def load_data(split_dir, per_class=None):
    images = []
    labels = []

    for label, class_name in enumerate(("cat", "dog")):
        files = sorted((Path(split_dir) / class_name).glob("*.jpg"))
        if per_class is not None:
            files = files[:per_class]

        for file in files:
            with Image.open(file) as image:
                image = image.convert("RGB").resize((64, 64))
                images.append(np.asarray(image, dtype=np.float32) / 255.0)
                labels.append(label)

    return np.array(images), np.array(labels, dtype=np.float32).reshape(-1, 1)


X_train, y_train = load_data("data/subset/train")
X_test, y_test = load_data("data/test")

print("Train:", X_train.shape, y_train.shape)
print("Test: ", X_test.shape, y_test.shape)
print("Pixel range:", X_train.min(), "to", X_train.max())

def relu(x):
    return np.maximum(0.0, x)


def conv2d(image, kernel):
    height, width, channels = image.shape
    kernel_height, kernel_width, kernel_channels = kernel.shape
    assert channels == kernel_channels

    output = np.zeros(
        (height - kernel_height + 1, width - kernel_width + 1),
        dtype=np.float32,
    )

    for row in range(output.shape[0]):
        for col in range(output.shape[1]):
            region = image[
                row : row + kernel_height,
                col : col + kernel_width,
                :
            ]
            output[row, col] = np.sum(region * kernel)

    return output


def maxpool(feature_map, size=2):
    height, width = feature_map.shape
    pooled_height = height // size
    pooled_width = width // size
    output = np.zeros((pooled_height, pooled_width), dtype=np.float32)

    for row in range(pooled_height):
        for col in range(pooled_width):
            region = feature_map[
                row * size : (row + 1) * size,
                col * size : (col + 1) * size
            ]
            output[row, col] = np.max(region)

    return output

kernel = np.random.default_rng(0).normal(0, 0.1, (3, 3, 3)).astype(np.float32)

conv = conv2d(X_train[0], kernel)
pooled = maxpool(relu(conv))
flat = pooled.flatten()

print("Convolution:", conv.shape)
print("Pooled:", pooled.shape)
print("Flattened size:", flat.shape)

rng = np.random.default_rng(0)
conv_filter = rng.normal(0, 0.1, (3, 3, 3)).astype(np.float32)

W_fc = rng.normal(0, 0.01, (961, 1)).astype(np.float32)
b_fc = np.zeros((1,), dtype=np.float32)


def sigmoid(x):
    x = np.clip(x, -50, 50)
    return 1.0 / (1.0 + np.exp(-x))


def forward(image):
    conv = conv2d(image, conv_filter)
    activated = relu(conv)
    pooled = maxpool(activated)
    flat = pooled.reshape(1, -1)
    probability = sigmoid(flat @ W_fc + b_fc)
    return probability[0, 0], flat

probability, features = forward(X_train[0])
print("Feature shape:", features.shape)
print("Dog probability:", probability)

def train_model(X, y, epochs=5, learning_rate=0.01):
    global W_fc, b_fc
    rng = np.random.default_rng(42)
    for epoch in range(epochs):
        total_loss = 0.0

        indices = rng.permutation(len(X))
        for i in indices:
            image, label = X[i], y[i]
            probability, features = forward(image)
            target = float(label[0])

            # Binary cross-entropy, with clipping to avoid log(0).
            p = np.clip(probability, 1e-7, 1.0 - 1e-7)
            total_loss += -(target * np.log(p) + (1.0 - target) * np.log(1.0 - p))

            # For sigmoid + binary cross-entropy, dL/dz = p - target.
            dz = probability - target
            grad_W = features.T * dz
            grad_b = dz

            W_fc -= learning_rate * grad_W
            b_fc -= learning_rate * grad_b

        print(f"Epoch {epoch + 1}/{epochs} | Loss: {total_loss / len(X):.4f}")

def evaluate(X, y):
    start = time.perf_counter()
    correct = 0

    for image, label in zip(X, y):
        probability, _ = forward(image)
        prediction = int(probability >= 0.5)
        correct += prediction == int(label[0])

    elapsed = time.perf_counter() - start
    accuracy = correct / len(X)

    print(f"Test accuracy: {accuracy * 100:.2f}% ({correct}/{len(X)})")
    print(f"Test processing time: {elapsed:.3f} seconds")
    print(f"Average time per image: {elapsed / len(X):.4f} seconds")


#train_model(X_train, y_train, epochs=50, learning_rate=0.001)

training_start = time.perf_counter()

train_model(X_train, y_train, epochs=50, learning_rate=0.001)

training_elapsed = time.perf_counter() - training_start
print(f"Training time: {training_elapsed:.3f} seconds")

evaluate(X_test, y_test)
