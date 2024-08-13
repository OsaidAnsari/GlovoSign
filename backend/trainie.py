import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.metrics import accuracy_score, classification_report
from keras.models import Sequential
from keras.layers import Conv1D, Flatten, Dense, MaxPooling1D, Dropout, BatchNormalization
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt

# Load the dataset
csv_file = 'sensor_readings.csv'  # Replace with the path to your CSV file
df = pd.read_csv(csv_file)

# Check for NaN values
print(df.isna().sum())

# Fill NaN values for numeric columns with the median
df.fillna(df.median(numeric_only=True), inplace=True)

# Fill NaN values in the Label column with the most frequent value
df['Label'].fillna(df['Label'].mode()[0], inplace=True)

# Verify that there are no more NaN values
print(df.isna().sum())

# Select features and target
features = df[['ADS1115 (0x48) A0', 'ADS1115 (0x48) A1', 'ADS1115 (0x48) A3', 'ADS1115 (0x49) A0', 'ADS1115 (0x49) A1']]
target = df['Label']

# Encode the labels to integers
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(target)

# One-hot encode the labels
y = to_categorical(y)

# Split the data into training and testing sets (67% training, 33% testing)
X_train, X_test, y_train, y_test = train_test_split(features, y, test_size=0.33, random_state=42)

# Expand dimensions for Conv1D
X_train_cnn = np.expand_dims(X_train, axis=2)
X_test_cnn = np.expand_dims(X_test, axis=2)

# Build the CNN model
def build_cnn():
    model = Sequential()
    
    # Convolutional Layer 1
    model.add(Conv1D(16, 2, activation='relu', input_shape=(X_train.shape[1], 1)))
    model.add(MaxPooling1D(1))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))
    
    # Convolutional Layer 2
    model.add(Conv1D(32, 2, activation='relu'))
    model.add(MaxPooling1D(1))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))
    
    # Flatten and Dense Layers
    model.add(Flatten())
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(y.shape[1], activation='softmax'))
    
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Initialize and train the CNN model with early stopping
cnn = build_cnn()
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
history = cnn.fit(X_train_cnn, y_train, epochs=50, batch_size=32, verbose=1, validation_split=0.2, callbacks=[early_stopping])

# Make predictions
y_pred_cnn = cnn.predict(X_test_cnn)

# Convert predictions and actual labels from one-hot encoding
y_pred_labels = np.argmax(y_pred_cnn, axis=1)
y_test_labels = np.argmax(y_test, axis=1)

# Calculate the accuracy
accuracy = accuracy_score(y_test_labels, y_pred_labels)
print(f'Accuracy: {accuracy * 100:.2f}%')

# Print the classification report
print(classification_report(y_test_labels, y_pred_labels))

# Save the model
model_filename = 'cnn_model.h5'
cnn.save(model_filename)
print(f'Model saved to {model_filename}')

# Plot accuracy and loss
plt.figure(figsize=(12, 6))

# Plot training & validation accuracy values
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(['Train', 'Test'], loc='upper left')

# Plot training & validation loss values
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(['Train', 'Test'], loc='upper left')

plt.show()
