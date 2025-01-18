import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os
from imblearn.over_sampling import SMOTE

# File paths
DATA_PATH = "static/data/locations.csv"
MODEL_PATH = "static/models/location_recommender.pkl"

def train_model():
    """
    Trains a Random Forest model to recommend locations based on budget and other features.
    Saves the trained model to the specified path.
    """
    # Load the dataset
    print("Loading dataset...")
    data = pd.read_csv(DATA_PATH)

    # Preprocess the data
    print("Preprocessing dataset...")

    # Drop unnecessary columns if they exist
    if 'name' in data.columns:
        data = data.drop(columns=['name'])

    # Convert categorical columns to numeric using one-hot encoding
    X = pd.get_dummies(data.drop(columns=['affordability']), drop_first=True)
    y = data['affordability']

    # Handle class imbalance using SMOTE
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X, y)

    # Split the data into training and testing sets
    print("Splitting dataset into training and testing sets...")
    X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

    # Simplified hyperparameter tuning with GridSearchCV
    print("Tuning hyperparameters...")
    param_grid = {
        'n_estimators': [50],  # Reduced number of estimators
        'max_depth': [10],  # Limited depth
        'min_samples_split': [5],
        'min_samples_leaf': [2],
        'class_weight': ['balanced']  # Adjusted class weights
    }
    grid_search = GridSearchCV(
        estimator=RandomForestClassifier(random_state=42),
        param_grid=param_grid,
        cv=3,
        scoring='accuracy',
        n_jobs=1  # Single-threaded for less CPU usage
    )
    grid_search.fit(X_train, y_train)
    print("Best parameters found: ", grid_search.best_params_)

    # Train the best model
    print("Training the model...")
    best_model = grid_search.best_estimator_
    best_model.fit(X_train, y_train)

    # Evaluate the model
    print("Evaluating the model...")
    y_pred = best_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model Accuracy: {accuracy:.2f}")
    print(classification_report(y_test, y_pred))

    # Save the trained model
    print("Saving the model...")
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump(best_model, MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")

if __name__ == "__main__":
    try:
        train_model()
    except Exception as e:
        print(f"Error during training: {e}")
