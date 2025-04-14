
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.metrics import accuracy_score

model = load_model("traffic_model.h5")

X_val = np.load("X_val.npy")
y_val = np.load("y_val.npy")

predictions = model.predict(X_val)
pred_labels = np.argmax(predictions, axis=1)
true_labels = np.argmax(y_val, axis=1)

accuracy = accuracy_score(true_labels, pred_labels)
print("Validation Accuracy:", accuracy)
