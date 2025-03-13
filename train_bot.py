import json
import pickle
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from nltk.tokenize import word_tokenize
from sklearn.preprocessing import LabelEncoder

# Load dataset
with open("agri_intents.json", encoding="utf-8-sig") as file:

    data = json.load(file)

words = []
classes = []
documents = []
ignore_words = ["?", "!"]

# Data preprocessing
for intent in data["intents"]:
    for pattern in intent["patterns"]:
        word_list = word_tokenize(pattern)
        words.extend(word_list)
        documents.append((word_list, intent["tag"]))
        if intent["tag"] not in classes:
            classes.append(intent["tag"])

words = sorted(set(words))
classes = sorted(set(classes))

# Save preprocessed data
pickle.dump(words, open("words.pkl", "wb"))
pickle.dump(classes, open("classes.pkl", "wb"))

# Convert data into training format
training = []
output_empty = [0] * len(classes)

for document in documents:
    bag = [1 if word in document[0] else 0 for word in words]
    output_row = list(output_empty)
    output_row[classes.index(document[1])] = 1
    training.append([bag, output_row])

training = np.array(training, dtype=object)
train_x = np.array([item[0] for item in training])
train_y = np.array([item[1] for item in training])

# Build a neural network model
model = Sequential([
    Dense(128, input_shape=(len(train_x[0]),), activation="relu"),
    Dropout(0.5),
    Dense(64, activation="relu"),
    Dropout(0.5),
    Dense(len(train_y[0]), activation="softmax")
])

model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

# Train the model
model.fit(train_x, train_y, epochs=200, batch_size=5, verbose=1)

# Save the trained model
model.save("chatbot_model.h5")

print("Training complete. Model saved.")
