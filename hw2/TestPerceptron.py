import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from Perceptron import Perceptron

df = pd.read_csv("archive/Iris.csv")
df = df[df["Species"] != "Iris-virginica"]

x = df[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']].values
y = df["Species"].values

y = np.where(y == "Iris-setosa", -1, 1)

ppn = Perceptron(eta=0.01, n_iter=10)
ppn.fit(x, y)

y_pred = ppn.predict(x)

accuracy = np.mean(y_pred == y)
print("Predictions:", y_pred)
print("True labels:", y)
print(f"Accuracy: {accuracy*100:.2f}%")

plt.plot(range(1, len(ppn.errors_) + 1), ppn.errors_, marker='o')
plt.xlabel('Epochs')
plt.ylabel('Number of misclassifications')
plt.title('Perceptron Learning Progress')
plt.show()
