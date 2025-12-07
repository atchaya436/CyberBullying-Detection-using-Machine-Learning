import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
import joblib
from sklearn.feature_extraction import text

# 1. Load the dataset
print("Loading dataset...")
try:
    df = pd.read_csv('cyberbullying_data.csv')
except FileNotFoundError:
    print("❌ Error: 'cyberbullying_data.csv' not found.")
    exit()

# 2. Simplify Labels (Bullying vs Non-Bullying)
def simplify_label(label):
    if label == 'not_cyberbullying':
        return 'non-bullying'
    else:
        return 'bullying'

print("Preparing data...")
df['cyberbullying_type'] = df['cyberbullying_type'].apply(simplify_label)

X = df['tweet_text']
y = df['cyberbullying_type']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Build Faster Pipeline (Logistic Regression)
# Logistic Regression is much faster than SVM for large text datasets
custom_stop_words = [
    "hello", "hi", "hey", "dear", "thanks", "good",  
    "school", "class", "student", "teacher", "udk","facebook", "twitter", "youtube", "video", "channel",         
    "mkr", "rt", "amp", "katandandre", "gud" ,"idk","tq","stop", "talk", "support", "report", "block", "bully", "bullied"            
    "u", "ur", "im", "dont", "lol", "ok", "okay","okiee","thx", "pls", "wanna", "gonna",     
]
my_stop_words = list(text.ENGLISH_STOP_WORDS.union(custom_stop_words))

# 3. Update the Pipeline
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(stop_words=my_stop_words, max_features=5000)),
    ('classifier', LogisticRegression(max_iter=1000)) 
])


# 4. Train
print("Training the model... (This should happen in seconds now)")
pipeline.fit(X_train, y_train)

# 5. Evaluate
print("Testing model accuracy...")
predictions = pipeline.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
print(f"✅ Model Accuracy: {accuracy * 100:.2f}%")

# 6. Save
joblib.dump(pipeline, 'cyberbullying_model.pkl')
print("🎉 Success! Model saved as 'cyberbullying_model.pkl'")