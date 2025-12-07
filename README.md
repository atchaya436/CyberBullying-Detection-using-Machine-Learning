# 🛡️ Cyberbullying Detection using Machine Learning

A **Machine Learning Application** that detects toxic language and cyberbullying in real-time. The system uses a trained Logistic Regression model served via a **FastAPI** backend and is accessible through a **Telegram Bot** interface hosted on the cloud.



[Image of machine learning classification workflow]


## 🚀 Project Overview

Social media moderation is a massive challenge. This project automates the detection of harmful content using Natural Language Processing (NLP). 

* **The Brain:** An ML model trained on a dataset of **47,000+ tweets**.
* **The Body:** A high-performance **FastAPI** server that handles requests.
* **The Interface:** A **Telegram Bot** that users can chat with directly from their phones.
* **The Home:** Hosted 24/7 on **Render Cloud**, kept awake by UptimeRobot.

## ✨ Key Features

* **Real-Time Detection:** Instantly flags bullying messages with a high-confidence warning.
* **Smart Filtering:** Includes a custom "Safe List" to prevent False Positives on greetings (e.g., "Hello") and command messages.
* **Probability Scores:** Calculates a confidence score (e.g., "95% sure this is toxic") before taking action.
* **Cloud Hosted:** Uses Webhooks to ensure the bot responds instantly without needing a local server running.
* **Scalable API:** The backend is built on FastAPI, ready to be integrated into Unity games, Discord bots, or web apps.

## 🛠️ Tech Stack

* **Language:** Python 3.9+
* **Machine Learning:** Scikit-Learn (Logistic Regression, TF-IDF Vectorizer)
* **Data Processing:** Pandas, NumPy, RegEx
* **Backend API:** FastAPI, Uvicorn
* **Bot Integration:** Telegram Bot API (Webhooks), HTTPX (Async requests)
* **Deployment:** Render (Cloud PaaS), UptimeRobot

## 📂 Project Structure

```bash
Cyberbullying-Detector/
│
├── api.py                  # The Main Server (FastAPI + Bot Logic)
├── train_model.py          # Script to train the ML model
├── cyberbullying_model.pkl # The saved 'Brain' (Trained Model)
├── requirements.txt        # List of dependencies for Cloud
├── cyberbullying_data.csv  # Dataset (Kaggle)
└── README.md               # Project Documentation
