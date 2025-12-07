# 🛡️ Cyberbullying Detection using Machine Learning

An end-to-end **Machine Learning NLP Project** that identifies toxic behavior and cyberbullying in text messages. The model is trained on social media data and deployed via a FastAPI backend to moderate conversations in real-time on Telegram.

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

## 🧠 Machine Learning Approach

This project focuses on **Natural Language Processing (NLP)** to classify text into `bullying` or `non-bullying` categories.

### 1. The Dataset
* **Source:** Kaggle Dataset of ~47,000 tweets.
* **Classes:** Balanced dataset containing roughly 80% bullying and 20% non-bullying content (handled via class weighting/probability thresholds).
* **Labels:** Cyberbullying types include ethnicity, gender, religion, and age-based harassment.

### 2. Data Preprocessing Pipeline
Raw text data is noisy. The following cleaning steps were implemented to improve model accuracy:
* **Tokenization:** Breaking sentences into individual words.
* **Normalization:** Converting all text to lowercase.
* **Noise Removal:** stripping special characters, URLs, and punctuation using Regex.
* **Custom Stop-Word Removal:**
* Standard English stop words (e.g., "the", "is") were removed.
* **Domain-Specific Optimization:** Words like *"Hello"*, *"School"*, and *"High"* were manually added to the exclusion list to prevent **False Positives** (innocent messages being flagged as toxic).

### 3. Feature Engineering 
To convert text into numerical data for the model, I used **TF-IDF (Term Frequency-Inverse Document Frequency)**.
* **Why TF-IDF?** Unlike simple word counts, TF-IDF reduces the weight of common words that appear everywhere and increases the weight of rare, significant words (like specific slurs or aggressive verbs).
* **Configuration:** Limited to the top 5,000 features to optimize performance and speed.

### 4. Model Architecture: Logistic Regression
The core classification engine is a **Logistic Regression** model.
* **Reason for Choice:** It offers an excellent balance between speed and accuracy for binary text classification tasks. It provides **probability scores** (Confidence %), allowing the system to ignore low-confidence predictions.
* **Performance:** The model achieves high accuracy in distinguishing between safe conversational text and aggressive harassment.

## 🏗️ System Architecture (Deployment)

While the core is ML, the model is wrapped in a full-stack application to make it usable in the real world:

* **Backend API:** Built with **FastAPI** to serve predictions via HTTP endpoints.
* **The "Bouncer" Logic:** A pre-processing layer in the API that instantly filters known safe words (e.g., greetings) and commands before they reach the model, reducing computational load.
* **Deployment:** Hosted on **Render Cloud** with a Webhook integration, allowing the model to sleep when inactive and wake up instantly for new requests.
* **Interface:** Integrated with a **Telegram Bot** for real-time user interaction.

## 🛠️ Tech Stack

* **Machine Learning:** Scikit-Learn, Pandas, NumPy, Joblib
* **NLP:** TF-IDF Vectorizer, Regex
* **Web Framework:** FastAPI, Uvicorn
* **Cloud & DevOps:** Render, UptimeRobot, Git
* **Client:** Telegram Bot API

**👨‍💻 Author**
[Atchaya K A]

