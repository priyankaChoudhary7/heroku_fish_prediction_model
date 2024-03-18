import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib

# Load your dataset
data = pd.read_csv("data/Fish.csv")

# Prepare features and target
X = data[['Weight', 'Length1', 'Height', 'Width']]
y = data['Species']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a RandomForestClassifier (or any other model of your choice)
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Save the trained model
joblib.dump(model, "fish_model.pkl")