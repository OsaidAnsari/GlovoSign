import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import joblib

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

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

# Initialize the Decision Tree Classifier
clf = DecisionTreeClassifier()

# Train the classifier
clf.fit(X_train, y_train)

# Make predictions
y_pred = clf.predict(X_test)

# Calculate the accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy * 100:.2f}%')

# Optional: Print the classification report
from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))

# Save the model
model_filename = 'decision_tree_model.joblib'
joblib.dump(clf, model_filename)
print(f'Model saved to {model_filename}')
