import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt

# Load the red wine dataset
red_wine_data = pd.read_csv("winequality-red.csv", sep=";")

# Load the white wine dataset
white_wine_data = pd.read_csv("winequality-white.csv", sep=";")

# Add a 'wine_type' column to distinguish between red and white wines
red_wine_data['wine_type'] = 'red'
white_wine_data['wine_type'] = 'white'

# Combine the datasets into a single DataFrame
combined_data = pd.concat([red_wine_data, white_wine_data])

# Encode the 'wine_type' column to numeric values
le = LabelEncoder()
combined_data['wine_type'] = le.fit_transform(combined_data['wine_type'])

# Define the features (X) and target variable (y)
X = combined_data.drop('quality', axis=1)
y = combined_data['quality']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features (excluding 'wine_type')
scaler = StandardScaler()
X_train[['wine_type']] = scaler.fit_transform(X_train[['wine_type']])
X_test[['wine_type']] = scaler.transform(X_test[['wine_type']])

# Build a Random Forest classifier
clf = RandomForestClassifier(random_state=42)
clf.fit(X_train, y_train)

# Make predictions
y_pred = clf.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
confusion = confusion_matrix(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred)

# Feature importances from the Random Forest model
feature_importances = clf.feature_importances_

# Get the feature names
feature_names = X.columns

# Sort feature importances in descending order
sorted_idx = np.argsort(feature_importances)[::-1]

# Print the top 5 most important features
print("Top 5 Most Important Features:")
for i in range(5):
    print(f"{feature_names[sorted_idx[i]]}: {feature_importances[sorted_idx[i]]:.4f}")

# Print model evaluation metrics
print(f"Accuracy: {accuracy:.4f}")
print("Confusion Matrix:")
print(confusion)
print("Classification Report:")
print(classification_rep)

# Plot feature importances
plt.figure(figsize=(10, 6))
plt.title("Feature Importances")
plt.bar(range(len(sorted_idx)), feature_importances[sorted_idx], align="center")
plt.xticks(range(len(sorted_idx)), [feature_names[i] for i in sorted_idx], rotation=45)
plt.xlabel("Feature")
plt.ylabel("Importance")
plt.tight_layout()
plt.show()