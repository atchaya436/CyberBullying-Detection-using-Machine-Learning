from fastapi import FastAPI
from pydantic import BaseModel
import joblib

# 1. Initialize the App
app = FastAPI()

# 2. Load your trained model
# We load the file you just created
model = joblib.load('cyberbullying_model.pkl')

# 3. Define the input format (we expect a JSON with a "text" field)
class Message(BaseModel):
    text: str

# 4. Create the API Route
@app.post("/predict")
def predict_bullying(message: Message):
    # Get the text
    user_text = message.text
    
    # Predict
    prediction = model.predict([user_text])[0]
    probs = model.predict_proba([user_text])
    confidence = probs.max()

    # --- THE FIX IS HERE ---
    # If confidence is low (less than 70%), assume it is SAFE
    if prediction == "bullying" and confidence < 0.70:
        prediction = "non-bullying"
        status = "SAFE"
    else:
        status = "FLAGGED" if prediction == "bullying" else "SAFE"
        
    return {
        "text": user_text,
        "prediction": prediction,
        "confidence": float(confidence),
        "status": status
    }

# 5. Simple root route to check if server is running
@app.get("/")
def home():
    return {"message": "Cyberbullying Detection API is Running!"}