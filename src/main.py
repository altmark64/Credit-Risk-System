from fastapi import FastAPI, HTTPException
import joblib
import pandas as pd

app = FastAPI(
    title="Credit Risk Scoring API",
    description="An AI-powered backend for predicting loan defaults.",
    version="1.0.0"
)

# 1. Load BOTH the Brain and the Blueprint (Feature List)
try:
    model = joblib.load('model_artifacts/random_forest_model.pkl')
    expected_features = joblib.load('model_artifacts/model_features.pkl')
    print("Model and Feature Blueprint loaded successfully!")
except Exception as e:
    print(f"Error loading model artifacts: {e}")

@app.get("/")
def health_check():
    return {"status": "online", "message": "Credit Risk API is active and listening."}

@app.post("/predict")
def predict_risk(application_data: dict):
    try:
        # 2. Convert incoming raw JSON into a DataFrame
        input_df = pd.DataFrame([application_data])
        
        # 3. Apply One-Hot Encoding to the incoming data
        input_df = pd.get_dummies(input_df)
        
        # 4. THE MAGIC TRICK: Reindex the columns to perfectly match the training data.
        # This adds missing columns and sets their value to 0, ensuring the shape is 100% correct.
        input_df = input_df.reindex(columns=expected_features, fill_value=0)
        
        # 5. Ask the AI for a prediction
        prediction_array = model.predict(input_df)
        prediction_value = int(prediction_array[0])
        
        # 6. Translate the math into plain English
        risk_label = "High Risk (Likely to Default)" if prediction_value == 1 else "Low Risk (Likely to Pay)"
        
        return {
            "status": "success",
            "prediction_code": prediction_value,
            "risk_assessment": risk_label
        }
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction failed: {str(e)}")