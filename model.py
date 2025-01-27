import joblib
import os

# Define paths to the saved model and label encoders
MODEL_PATH = 'static/models/location_recommender_model.pkl'
ENCODERS_PATH = 'static/models/label_encoders.pkl'

def load_model():
    """
    This function loads the trained model and label encoders from disk.
    
    Returns:
        model: the trained machine learning model
        label_encoders: a dictionary containing label encoders
    """
    try:
        # Check if the model and encoder files exist
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")
        
        if not os.path.exists(ENCODERS_PATH):
            raise FileNotFoundError(f"Label encoders file not found at {ENCODERS_PATH}")
        
        # Load the model and label encoders
        model = joblib.load(MODEL_PATH)
        label_encoders = joblib.load(ENCODERS_PATH)
        
        return model, label_encoders
    
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return None, None
    except Exception as e:
        print(f"Error loading model or label encoders: {e}")
        return None, None