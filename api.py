from fastapi import FastAPI, Request
from pydantic import BaseModel
import joblib
import uvicorn
import httpx
import os

# 1. SETUP
TOKEN = os.environ.get("BOT_TOKEN")

if not TOKEN:
    raise ValueError("No BOT_TOKEN found! Did you set it in Render?")

TELEGRAM_API_URL = f"https://api.telegram.org/bot{TOKEN}/sendMessage"

# --- THE FIX: SAFE WORD LIST ---
SAFE_WORDS = ["hello", "hi", "hey", "start", "test", "thanks", "good morning", "good night", "ok", "okay","dear"
    "school", "class", "student", "teacher", "udk","facebook", "twitter", "youtube", "video", "channel",         
    "mkr", "rt", "amp", "katandandre", "gud" ,"idk","tq","stop", "talk", "support", "report", "block", "bully", "bullied"            
    "u", "ur", "im", "dont", "lol", "ok", "okay","okiee","thx", "pls", "wanna", "gonna",]

# 2. DATA MODELS
class Message(BaseModel):
    text: str

# 3. PREDICTION ENGINE
@app.post("/predict")
def predict_bullying(message: Message):
    user_text = message.text.lower().strip()
    
    # A. THE BOUNCER CHECK (Immediate Safe)
    if user_text in SAFE_WORDS or user_text.startswith('/'):
        return {
            "text": message.text,
            "prediction": "non-bullying",
            "confidence": 1.0,
            "status": "SAFE"
        }

    # B. AI CHECK
    prediction = model.predict([user_text])[0]
    probs = model.predict_proba([user_text])
    confidence = probs.max()
    
    status = "FLAGGED" if prediction == "bullying" else "SAFE"
    
    return {
        "text": message.text,
        "prediction": prediction,
        "confidence": float(confidence),
        "status": status
    }

# 4. TELEGRAM WEBHOOK
@app.post("/webhook")
async def telegram_webhook(request: Request):
    data = await request.json()
    
    if "message" in data and "text" in data["message"]:
        chat_id = data["message"]["chat"]["id"]
        user_text = data["message"]["text"]
        clean_text = user_text.lower().strip()
        
        # --- IGNORE COMMANDS & GREETINGS ---
        if clean_text.startswith('/') or clean_text in SAFE_WORDS:
            return {"status": "ignored"}
        # -----------------------------------
        
        # Analyze with Model
        prediction = model.predict([clean_text])[0]
        
        if prediction == "bullying":
            probs = model.predict_proba([clean_text])
            confidence = probs.max()
            
            # Only block if confidence is high (> 70%) to reduce false positives
            if confidence > 0.70:
                reply_text = f"⚠️ Warning: Toxic content detected.\nConfidence: {confidence*100:.1f}%"
                
                async with httpx.AsyncClient() as client:
                    await client.post(TELEGRAM_API_URL, json={
                        "chat_id": chat_id,
                        "text": reply_text
                    })
                
    return {"status": "ok"}

@app.get("/")
def home():

    return {"message": "Cyberbullying Detection System is Live!"}
