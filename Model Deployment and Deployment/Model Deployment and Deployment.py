from fastapi import FastAPI, HTTPException
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

app = FastAPI()

# Read the dataset
df = pd.read_csv("dhaka_road_system.csv")

# Handle non-numeric features
# For simplicity, let's drop non-numeric features for now
df_numeric = df.select_dtypes(include=['number'])

# Split the data into train and test sets
X = df_numeric.drop(columns=["Class"])
y = df_numeric["Class"]

# Encode categorical target variable if applicable
le = LabelEncoder()
y = le.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the machine learning model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Model Accuracy:", accuracy)

# Define API endpoints
@app.get("/")
def read_root():
    return {"message": "Welcome to the road system prediction API!"}

@app.post("/predict")
def predict_class(features: dict):
    if not features:
        raise HTTPException(status_code=400, detail="No features provided.")
    features_df = pd.DataFrame(features, index=[0])
    features_numeric = features_df.select_dtypes(include=['number'])
    if features_numeric.empty:
        raise HTTPException(status_code=400, detail="No numeric features provided.")
    prediction = model.predict(features_numeric)
    predicted_class = le.inverse_transform(prediction)
    return {"predicted_class": predicted_class.tolist()}

# Run the FastAPI application
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
