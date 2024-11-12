import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score
import pickle

# Load the data from the CSV file
df = pd.read_csv('data.csv')

# Drop all columns that are not being used for prediction (keeping only relevant columns)
df = df[['Kills', 'Wins', 'Damage_Dealt', 'Time_Survived', 'Rank']]

# Check if the data loaded correctly
print(df.head())

# Separate features (X) and target (y)
X = df[['Kills', 'Wins', 'Damage_Dealt', 'Time_Survived']]
y = df['Rank']

# Encode target labels (Rank) into numeric values
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.3, random_state=42)

# Define parameter grids for each model
param_grids = {
    'LogisticRegression': {
        'logisticregression__C': [0.1, 1, 10, 100],
        'logisticregression__solver': ['liblinear', 'lbfgs'],
        'logisticregression__max_iter': [100, 200, 500]
    },
    'RandomForestClassifier': {
        'randomforestclassifier__n_estimators': [50, 100, 200],
        'randomforestclassifier__max_depth': [None, 10, 20, 30],
        'randomforestclassifier__min_samples_split': [2, 5, 10]
    },
    'SVC': {
        'svc__C': [0.1, 1, 10],
        'svc__kernel': ['linear', 'rbf'],
        'svc__gamma': ['scale', 'auto']
    }
}

# Define models
models = {
    'LogisticRegression': make_pipeline(LogisticRegression(max_iter=1000000)),
    'RandomForestClassifier': make_pipeline(RandomForestClassifier(random_state=42)),
    'SVC': make_pipeline(SVC())
}

# Track the best model and accuracy
best_model = None
best_accuracy = 0
best_params = {}

# Perform grid search for each model
for model_name, model in models.items():
    print(f"Running GridSearchCV for {model_name}...")
    grid_search = GridSearchCV(model, param_grids[model_name], cv=5, scoring='accuracy', n_jobs=-1, verbose=1)
    grid_search.fit(X_train, y_train)
    
    # Get accuracy for this model
    y_pred = grid_search.best_estimator_.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"{model_name} accuracy: {accuracy}")
    
    # Update best model if this one is better
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_model = grid_search.best_estimator_
        best_params = grid_search.best_params_

# Print the best model details
print("Best model:", best_model)
print("Best accuracy on test set:", best_accuracy)
print("Best parameters:", best_params)

# Save the best model and label encoder using pickle
with open('best_rank_prediction_pipeline.pkl', 'wb') as f:
    pickle.dump(best_model, f)

with open('label_encoder.pkl', 'wb') as f:
    pickle.dump(label_encoder, f)

print("Best model trained, accuracy calculated, and saved successfully!")
