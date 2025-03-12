import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.metrics import classification_report, ConfusionMatrixDisplay, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

#Step 1: Data Collection & Preprocessing

# Load dataset
df = pd.read_csv('train.csv')

df.head()

# Select relevant features
selected_features = ['Survived', 'Pclass', 'Sex', 'Age', 'Fare', 'Embarked']
df = df[selected_features]

df.info()

# Handle missing values
df['Age'].fillna(df['Age'].median(), inplace=True)  # Fill missing age with median
df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)  # Fill missing embarked with mode

# Convert categorical features to numerical using label encoding
df['Sex'] = (df['Sex'] == 'male').astype(int)  # Convert 'male' to 1, 'female' to 0

# Convert 'Embarked' into numerical values using one hot encoding
df['Embarked_C'] = (df['Embarked'] == 'C').astype(int)
df['Embarked_Q'] = (df['Embarked'] == 'Q').astype(int)
df['Embarked_S'] = (df['Embarked'] == 'S').astype(int)

# Drop Embarked Column
df.drop('Embarked', axis=1, inplace=True)

df.info()

# Step 2: Feature Selection & Engineering

# Standardize numerical features
scaler = StandardScaler()
df[['Age', 'Fare']] = scaler.fit_transform(df[['Age', 'Fare']])

# Define features and target
X = df[['Age', 'Embarked_C', 'Embarked_Q', 'Embarked_S', 'Pclass', 'Sex', 'Fare']]
y = df['Survived']

# Step 3: Train & Evaluate the Model

# Split Data into Training and Test Sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize SMOTE instance
smote = SMOTE(random_state=42)

# Apply SMOTE to the training data
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# Define the model
model = RandomForestClassifier(random_state=42)

# Set up the parameter grid for GridSearchCV
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10], 'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2', 0.5, 1],
    'bootstrap': [True, False]
}

# Set up GridSearchCV to find the best parameters
grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2)

# Fit the model
grid_search.fit(X_train_resampled, y_train_resampled)

# Get the best parameters and best score
best_params = grid_search.best_params_
best_score = grid_search.best_score_

print("Best Parameters:", best_params)
print("Best Score:", best_score)

# Get the best model
best_model = grid_search.best_estimator_

# Make Predictions
y_pred = best_model.predict(X_test)

# Evaluate Model Performance
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)
print("Confusion Matrix:\n", conf_matrix)

ConfusionMatrixDisplay.from_estimator(best_model, X_test, y_test)
plt.show()

# Save Model and Scaler
with open('titanic_model.pkl', 'wb') as file:
    pickle.dump(best_model, file)

with open('scaler.pkl', 'wb') as file:
    pickle.dump(scaler, file)

# Load the model
with open('titanic_model.pkl', 'rb') as file:
    loaded_model = pickle.load(file)

# Load the scaler
with open('scaler.pkl', 'rb') as file:
    loaded_scaler = pickle.load(file)
