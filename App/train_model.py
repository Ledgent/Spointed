import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt

# File paths
DATA_PATH = "static/data/locations.csv"  # Ensure this is correct
MODEL_PATH = "static/models/location_recommender.pkl"
FEATURE_NAMES_PATH = "static/models/feature_names.pkl"

def train_model():
    """
    Trains a Random Forest model to recommend restaurant locations.
    Saves the trained model and feature names to files.
    """
    # Load the dataset
    print("Loading dataset...")
    data = pd.read_csv(DATA_PATH)

    # Preprocess the data
    print("Preprocessing dataset...")
    if 'name' in data.columns:
        data = data.drop(columns=['name'])  # Drop irrelevant columns

    # Convert categorical columns to numeric using one-hot encoding
    X = pd.get_dummies(data.drop(columns=['affordability']), drop_first=True)
    y = data['affordability']

    # Handle class imbalance using SMOTE
    print("Handling class imbalance using SMOTE...")
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X, y)

    # Split the data into training and testing sets
    print("Splitting dataset into training and testing sets...")
    X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

    # Hyperparameter tuning with GridSearchCV
    print("Tuning hyperparameters...")
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [None, 10],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2],
        'class_weight': ['balanced', None]
    }
    grid_search = GridSearchCV(
        estimator=RandomForestClassifier(random_state=42),
        param_grid=param_grid,
        cv=3,
        scoring='accuracy',
        n_jobs=-1
    )
    grid_search.fit(X_train, y_train)
    print("Best parameters found: ", grid_search.best_params_)

    # Train the best model
    print("Training the best model...")
    best_model = grid_search.best_estimator_
    best_model.fit(X_train, y_train)

    # Evaluate the model
    print("Evaluating the model...")
    y_pred = best_model.predict(X_test)
    print(f"Model Accuracy: {accuracy_score(y_test, y_pred):.2f}")
    print(classification_report(y_test, y_pred))

    # Feature importance plot
    print("Plotting feature importance...")
    feature_importances = best_model.feature_importances_
    features = X.columns
    sorted_idx = feature_importances.argsort()

    plt.figure(figsize=(10, 6))
    plt.barh(features[sorted_idx][-10:], feature_importances[sorted_idx][-10:])
    plt.xlabel('Feature Importance')
    plt.title('Feature Importance in the Model')
    plt.show()

    # Save the model and feature names
    print("Saving the model and feature names...")
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump(best_model, MODEL_PATH)
    joblib.dump(list(X.columns), FEATURE_NAMES_PATH)
    print(f"Model saved to {MODEL_PATH}")
    print(f"Feature names saved to {FEATURE_NAMES_PATH}")

if __name__ == "__main__":
    try:
        train_model()
    except Exception as e:
        print(f"Error during training: {e}")
