import pickle
import numpy as np
import pandas as pd


#axis = 1 -> across column
def softmax(z):
    z = z - np.max(z, axis=1, keepdims=True)
    e_pow_z = np.exp(z)
    return e_pow_z / np.sum(e_pow_z, axis=1, keepdims=True)

def forward(x, w, b):
    z = np.dot(x, w) + b
    return softmax(z)

def compute_gradients(x, y_binary, probs):
    batch_size = x.shape[0]
    dz = probs - y_binary
    dw = np.dot(x.T, dz) / batch_size
    db = np.sum(dz, axis=0) / batch_size
    return dw, db


def train(x, y_binary, lr=0.01, n_epochs=100, batch_size=32):

    w = np.zeros((784, 10))
    b = np.zeros(10)

    for epoch in range(n_epochs):
        indices = np.arange(x.shape[0])
        np.random.shuffle(indices)
        X_shuffled = x[indices]
        y_shuffled = y_binary[indices]

        for i in range(0, x.shape[0], batch_size):
            X_batch = X_shuffled[i:i+batch_size]
            y_batch = y_shuffled[i:i+batch_size]

            probs = forward(X_batch, w, b)
            dw, db = compute_gradients(X_batch, y_batch, probs)


            w -= lr * dw
            b -= lr * db

        if (epoch + 1) % 10 == 0 or epoch == 0:
            preds = np.argmax(forward(x, w, b), axis=1)
            acc = np.mean(preds == np.argmax(y_binary, axis=1))
            print(f"Epoch {epoch + 1}/{n_epochs} - Training accuracy: {acc:.4f}")

    return w, b


def predict(X, W, b):
    probs = forward(X, W, b)
    return np.argmax(probs, axis=1)


def main():
    train_file = "fii-nn-2025-homework-2/extended_mnist_train.pkl"
    test_file = "fii-nn-2025-homework-2/extended_mnist_test.pkl"

    with open(train_file, "rb") as fp:
        train_data_raw = pickle.load(fp)
    with open(test_file, "rb") as fp:
        test_data_raw = pickle.load(fp)

    train_data = np.array([image.flatten() / 255.0 for image, label in train_data_raw])
    train_labels = np.array([label for image, label in train_data_raw])
    test_data = np.array([image.flatten() / 255.0 for image, label in test_data_raw])
    n_classes = 10

    #labels.shape[0] = rows
    y_train_binary = np.zeros((train_labels.shape[0], n_classes))
    for i in range(n_classes):
        y_train_binary[:, i] = np.where(train_labels == i, 1, 0)

    W, b = train(train_data, y_train_binary, lr=0.01, n_epochs=500)

    train_predictions = predict(train_data, W, b)
    train_acc = np.mean(train_predictions == train_labels)
    print(f"Training accuracy: {train_acc:.4f}")


    test_predictions = predict(test_data, W, b)

    predictions_csv = {
        "ID": [],
        "target": [],
    }

    for i, label in enumerate(test_predictions):
        predictions_csv["ID"].append(i)
        predictions_csv["target"].append(label)

    df = pd.DataFrame(predictions_csv)
    df.to_csv("submission.csv", index=False)

if __name__ == "__main__":
    main()
