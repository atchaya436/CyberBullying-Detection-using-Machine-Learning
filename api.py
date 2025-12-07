from fastapi import FastAPI, Request
from pydantic import BaseModel
import joblib
import uvicorn
import httpx
import os

# 1. SETUP
app = FastAPI()
model = joblib.load('cyberbullying_model.pkl')
TOKEN = "8264186073:AAGbAkBPHPiY3e2Tf5yid20bu4qSRIKJ4Cc"  
TELEGRAM_API_URL = f"https://api.telegram.org/bot{TOKEN}/sendMessage"

# 2. DATA MODELS
class Message(BaseModel):
    text: str

# 3. PREDICTION ENGINE (The Brain)
@app.post("/predict")
def predict_bullying(message: Message):
    user_text = message.text
    prediction = model.predict([user_text])[0]
    probs = model.predict_proba([user_text])
    confidence = probs.max()
    
    status = "FLAGGED" if prediction == "bullying" else "SAFE"
    
    return {
        "text": user_text,
        "prediction": prediction,
        "confidence": float(confidence),
        "status": status
    }

# 4. TELEGRAM WEBHOOK (The Bot Logic)
@app.post("/webhook")
async def telegram_webhook(request: Request):
    data = await request.json()
    
    # Check if it's a text message
    if "message" in data and "text" in data["message"]:
        chat_id = data["message"]["chat"]["id"]
        user_text = data["message"]["text"]
        
        # A. Analyze the text internally
        prediction = model.predict([user_text])[0]
        
        # B. If bullying, reply to the user
        if prediction == "bullying":
            probs = model.predict_proba([user_text])
            confidence = probs.max()
            
            reply_text = f"⚠️ Warning: Toxic content detected.\nConfidence: {confidence*100:.1f}%"
            
            # Send message back to Telegram
            async with httpx.AsyncClient() as client:
                await client.post(TELEGRAM_API_URL, json={
                    "chat_id": chat_id,
                    "text": reply_text
                })
                
    return {"status": "ok"}

@app.get("/")
def home():
    return {"message": "Cyberbullying Detection System is Live!"}