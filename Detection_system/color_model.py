import cv2
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import joblib

# Load the dataset of color features and labels
# Replace 'color_features.npy' and 'color_labels.npy' with your dataset files
color_features = np.load('color_features.npy')
color_labels = np.load('color_labels.npy')

# Encode color labels into numerical values
label_encoder = LabelEncoder()
color_labels_encoded = label_encoder.fit_transform(color_labels)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(color_features, color_labels_encoded, test_size=0.2, random_state=42)

# Create a k-nearest neighbors classifier
classifier = KNeighborsClassifier(n_neighbors=5)

# Train the classifier
classifier.fit(X_train, y_train)

# Evaluate the classifier on the testing set (optional)
accuracy = classifier.score(X_test, y_test)
print(f"Classifier Accuracy: {accuracy:.2f}")

# Save the trained classifier to a file
joblib.dump(classifier, 'color_classifier.pkl')