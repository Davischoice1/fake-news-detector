from flask import Flask, render_template, request
import joblib
import re
import string
import xgboost as xgb

app = Flask(__name__)

# Load model and vectorizer
model = xgb.XGBClassifier()
model.load_model("best_model.json")
vectorizer = joblib.load("vectorizer.pkl")

# Clean text
def wordopt(text):
    text = text.lower()
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'\\W', ' ', text)
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub(r'\n', ' ', text)
    text = re.sub(r'\w*\d\w*', '', text)
    return text.strip()

# Output label
def output_label(pred):
    return "Fake News" if pred == 1 else "Not Fake News"

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = ""
    confidence = ""
    news_input = ""
    
    if request.method == "POST":
        news_input = request.form["news"]
        cleaned = wordopt(news_input)
        vectorized = vectorizer.transform([cleaned])
        
        # Get prediction and probability
        pred_class = model.predict(vectorized)[0]
        pred_prob = model.predict_proba(vectorized)[0][pred_class]

        prediction = output_label(pred_class)
        confidence = f"{pred_prob * 100:.2f}%"  # convert to percentage string

    return render_template("index.html", prediction=prediction, confidence=confidence, news=news_input)

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=10000)
