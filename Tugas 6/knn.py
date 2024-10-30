import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix

# Load the dataset
data = pd.read_csv('CrabCopy.csv')

# Prepare features (X) and target (y)
X = data.drop(['Sex', 'Age'], axis=1)  # Remove Sex and Age columns
y = data['Sex']  # Use Sex as target variable

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Create and train the KNN model
k = 5  # number of neighbors
knn = KNeighborsClassifier(n_neighbors=k)
knn.fit(X_train_scaled, y_train)

# Make predictions
y_pred = knn.predict(X_test_scaled)

# Print the results
print("KNN Classification Results (k={}):\n".format(k))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Function to predict new data
def predict_crab_sex(features):
    features_scaled = scaler.transform([features])
    prediction = knn.predict(features_scaled)
    return prediction[0]

# Example usage:
if __name__ == "__main__":
    # Example of predicting a new crab's sex
    sample_crab = [1.4375, 1.175, 0.4125, 24.6357155, 12.3320325, 5.5848515, 6.747181]
    predicted_sex = predict_crab_sex(sample_crab)
    print("\nSample Prediction:")
    print("Predicted Sex:", predicted_sex)