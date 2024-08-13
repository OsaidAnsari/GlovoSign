
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from keras.models import Sequential
from keras.layers import Dense, Conv1D, Flatten, LSTM, Dropout
from keras.utils import to_categorical
import matplotlib.pyplot as plt
import seaborn as sns

# Load the data
file_path = "C:/Users/Muneer/Desktop/sensor_readings.csv"  # Path to your CSV file
data = pd.read_csv(file_path)

# Preprocessing
X = data[['ADS1115 (0x48) A0', 'ADS1115 (0x48) A1', 'ADS1115 (0x48) A3', 'ADS1115 (0x49) A0', 'ADS1115 (0x49) A1']]
y = data['Label']

# Encode labels
le = LabelEncoder()
y = le.fit_transform(y)
y = to_categorical(y)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# ANN model
def build_ann():
    model = Sequential()
    model.add(Dense(64, activation='relu', input_shape=(X_train.shape[1],)))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(y.shape[1], activation='softmax'))
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

ann = build_ann()
ann.fit(X_train, y_train, epochs=20, batch_size=32, verbose=1)
y_pred_ann = ann.predict(X_test)
cm_ann = confusion_matrix(y_test.argmax(axis=1), y_pred_ann.argmax(axis=1))


# CNN model
def build_cnn():
    model = Sequential()
    model.add(Conv1D(64, 2, activation='relu', input_shape=(X_train.shape[1], 1)))
    model.add(Flatten())
    model.add(Dense(64, activation='relu'))
    model.add(Dense(y.shape[1], activation='softmax'))
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

X_train_cnn = np.expand_dims(X_train, axis=2)
X_test_cnn = np.expand_dims(X_test, axis=2)
cnn = build_cnn()
cnn.fit(X_train_cnn, y_train, epochs=20, batch_size=32, verbose=1)
y_pred_cnn = cnn.predict(X_test_cnn)
cm_cnn = confusion_matrix(y_test.argmax(axis=1), y_pred_cnn.argmax(axis=1))


# KNN model
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train.argmax(axis=1))
y_pred_knn = knn.predict(X_test)
cm_knn = confusion_matrix(y_test.argmax(axis=1), y_pred_knn)


# LSTM model
def build_lstm():
    model = Sequential()
    model.add(LSTM(64, input_shape=(X_train.shape[1], 1)))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(y.shape[1], activation='softmax'))
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

lstm = build_lstm()
lstm.fit(X_train_cnn, y_train, epochs=20, batch_size=32, verbose=1)
y_pred_lstm = lstm.predict(X_test_cnn)
cm_lstm = confusion_matrix(y_test.argmax(axis=1), y_pred_lstm.argmax(axis=1))


# Plot confusion matrices
def plot_confusion_matrix(cm, title):
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(title)
    plt.ylabel('Actual Label')
    plt.xlabel('Predicted Label')
    plt.show()

plot_confusion_matrix(cm_ann, "ANN Confusion Matrix")
plot_confusion_matrix(cm_cnn, "CNN Confusion Matrix")
plot_confusion_matrix(cm_knn, "KNN Confusion Matrix")
plot_confusion_matrix(cm_lstm, "LSTM Confusion Matrix")