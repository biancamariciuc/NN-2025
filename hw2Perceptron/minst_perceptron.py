import pickle
import pandas as pd
import numpy as np
from Perceptron import Perceptron

train_file = "fii-nn-2025-homework-2/extended_mnist_train.pkl"
test_file = "fii-nn-2025-homework-2/extended_mnist_test.pkl"

with open(train_file, "rb") as fp:
    train = pickle.load(fp)

with open(test_file, "rb") as fp:
    test = pickle.load(fp)

def predict(X, perceptrons):
    scores = np.array([p.forward(X) for p in perceptrons])
    return np.argmax(scores, axis=0)

train_data = np.array([image.flatten() / 255.0 for image, label in train])
train_labels = np.array([label for image, label in train])
test_data = np.array([image.flatten() / 255.0 for image, label in test])

n_samples, n_features = train_data.shape

perceptrons = []
for digit in range(10):
    print(f"perceptron for digit {digit}")
    y_binary = np.where(train_labels == digit, 1, 0)
    p = Perceptron(n_features=n_features, lr=0.01, n_epochs=100)
    p.train(train_data, y_binary)
    perceptrons.append(p)

train_predictions = predict(train_data, perceptrons)
acc = np.mean(train_predictions == train_labels)
print(f"Training accuracy: {acc:.4f}")

test_predictions = predict(test_data, perceptrons)

predictions_csv = {
    "ID": [],
    "target": [],
}

for i, label in enumerate(test_predictions):
    predictions_csv["ID"].append(i)
    predictions_csv["target"].append(label)

df = pd.DataFrame(predictions_csv)
df.to_csv("submission.csv", index=False)
