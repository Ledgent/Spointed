"""
Optimized Restaurant Budget Prediction System with NLP and Reinforcement Learning
Author: Philipp Amana
Date: [July 17 2024]

Features:
1. Efficient data preprocessing with Pandas (no Dask dependency)
2. Text feature extraction using TF-IDF (limited dimensions)
3. Lightweight Gradient Boosted Models with reduced complexity
4. Multi-armed bandit implementation for model selection
5. Memory-efficient sparse matrix operations
6. Full compatibility with low-spec devices
"""

import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from scipy.sparse import hstack
import lightgbm as lgb
import xgboost as xgb
import joblib

# ========================
# 1. Data Preparation
# ========================

def convert_budget(budget_str):
    """Convert budget strings to numerical values"""
    budget_str = budget_str.replace(' ', '').replace('+', '').replace('₦', '').replace('NGN', '')
    if '-' in budget_str:
        low, high = budget_str.split('-')
    elif '+' in budget_str:
        low = budget_str.replace('+', '')
        high = str(float(low.replace('M', '')) * 1.5) + 'M'
    else:
        low = high = budget_str

    def convert_part(part):
        part = part.strip()
        if 'M' in part:
            return float(part.replace('M', '')) * 1e6
        elif 'k' in part:
            return float(part.replace('k', '')) * 1e3
        return float(part)

    return (convert_part(low) + convert_part(high)) / 2

# Load and process data
df = pd.read_csv('static/data/kaduna_north_restaurant_business_locations.csv')
df['budget'] = df['Budget'].apply(convert_budget)
df['text'] = df['Description'] + ' ' + df['Advice']  # Combine text features

# Select relevant columns
keep_cols = ['Location_Type', 'Foot_Traffic', 'Affordability', 'Competition',
             'Restaurant_Type', 'Business_Size', 'Category', 'Monthly_Rent_NGN',
             'text', 'budget']
df = df[keep_cols]

# ========================
# 2. Feature Engineering
# ========================

# Split data before feature engineering to prevent data leakage
X_train, X_test, y_train, y_test = train_test_split(
    df.drop('budget', axis=1), df['budget'], 
    test_size=0.2, random_state=42
)

# Text feature pipeline
text_transformer = Pipeline([
    ('tfidf', TfidfVectorizer(
        max_features=50,  # Keep dimensions low
        stop_words='english',
        ngram_range=(1, 2)
    ))
])

# Categorical and numerical features
categorical_cols = ['Location_Type', 'Foot_Traffic', 'Affordability', 
                   'Competition', 'Restaurant_Type', 'Business_Size', 'Category']
numerical_cols = ['Monthly_Rent_NGN']

preprocessor = ColumnTransformer([
    ('text', text_transformer, 'text'),
    ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols),
    ('num', StandardScaler(), numerical_cols)
], remainder='drop')

# Fit preprocessing on training data
X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)

# ========================
# 3. Model Training
# ========================

# Lightweight model configurations
models = {
    'LightGBM': lgb.LGBMRegressor(
        num_leaves=15,
        max_depth=3,
        n_estimators=50,
        learning_rate=0.1,
        verbose=-1
    ),
    'RandomForest': RandomForestRegressor(
        n_estimators=50,
        max_depth=5,
        random_state=42,
        n_jobs=-1
    ),
    'XGBoost': xgb.XGBRegressor(
        n_estimators=50,
        max_depth=3,
        learning_rate=0.1,
        tree_method='hist'  # Memory-efficient
    )
}

# Train all models
trained_models = {}
for name, model in models.items():
    model.fit(X_train_processed, y_train)
    trained_models[name] = model
    print(f"{name} trained")

# ========================
# 4. Reinforcement Learning (Bandit) Setup
# ========================

class BanditModelSelector:
    """Epsilon-greedy multi-armed bandit for model selection"""
    def __init__(self, models, epsilon=0.1, learning_rate=0.01):
        self.models = models
        self.epsilon = epsilon
        self.lr = learning_rate
        self.q_values = {name: 0.0 for name in models.keys()}
        self.n_updates = {name: 0 for name in models.keys()}
    
    def select_model(self):
        if np.random.random() < self.epsilon:
            return np.random.choice(list(self.models.keys()))
        else:
            return max(self.q_values, key=self.q_values.get)
    
    def update(self, model_name, reward):
        self.n_updates[model_name] += 1
        self.q_values[model_name] += self.lr * (reward - self.q_values[model_name])

# Initialize bandit with negative MSE as reward
bandit = BanditModelSelector(trained_models, epsilon=0.2)

# ========================
# 5. Evaluation & Bandit Training
# ========================

# Convert to dense arrays if sparse (required for some models)
if hasattr(X_test_processed, 'toarray'):
    X_test_dense = X_test_processed.toarray()
else:
    X_test_dense = X_test_processed

# Bandit learning loop
total_reward = 0
predictions = []
rewards = []

for i in range(len(X_test_dense)):
    # Select model
    selected_model = bandit.select_model()
    model = trained_models[selected_model]
    
    # Make prediction
    pred = model.predict(X_test_dense[i:i+1])
    actual = y_test.iloc[i]
    mse = mean_squared_error([actual], pred)
    
    # Calculate reward (negative MSE)
    reward = -mse
    total_reward += reward
    
    # Update bandit
    bandit.update(selected_model, reward)
    
    predictions.append(pred[0])
    rewards.append(reward)

# Final evaluation
bandit_rmse = np.sqrt(mean_squared_error(y_test, predictions))
best_single_model = max(trained_models, key=lambda k: bandit.q_values[k])

print("\n=== Bandit Performance ===")
print(f"Total Reward: {total_reward:.2f}")
print(f"Final Q-values: {bandit.q_values}")
print(f"Best Performing Model: {best_single_model}")
print(f"Bandit RMSE: {bandit_rmse:.2f}")

# Compare with individual models
print("\n=== Individual Model Performance ===")
for name, model in trained_models.items():
    pred = model.predict(X_test_processed)
    rmse = np.sqrt(mean_squared_error(y_test, pred))
    r2 = r2_score(y_test, pred)
    print(f"{name}:")
    print(f"  RMSE: {rmse:.2f}")
    print(f"  R²: {r2:.4f}")

# ========================
# 6. Saving System Components
# ========================

# Create directory structure
os.makedirs('static/models', exist_ok=True)

# Save components
joblib.dump(trained_models, 'static/models/model_suite.pkl')
joblib.dump(preprocessor, 'static/models/preprocessor.pkl')
joblib.dump(bandit, 'static/models/bandit_selector.pkl')

print("\nSystem training complete! All components saved.")