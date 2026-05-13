from flask import Flask, request, jsonify, render_template_string
import joblib
from sklearn.feature_extraction.text import CountVectorizer

import sys
sys.path.append("clean data")
from clean import clean_text

app = Flask(__name__)

model = joblib.load("BEST_MODEL_F1_trial.pkl")

classes = list(model.classes_)
print("MODEL CLASSES ORDER:", classes)

LABELS = {i: label for i, label in enumerate(classes)}

VOCAB_PATH = "representations/scheme3_vocab.txt"

with open(VOCAB_PATH, "r", encoding="utf-8") as f:
    vocab = [line.strip() for line in f]

vectorizer = CountVectorizer(vocabulary=vocab)

def preprocess(text):
    return clean_text(
        str(text),
        remove_special=True,
        clean_emojis=True,
        drop_emojis=False,
        lowercase=True,
        correct_spelling=False,
        lemmatize=False,
        remove_stops=True
    )

def predict_text(text):
    text = preprocess(text)
    X = vectorizer.transform([text])

    pred = model.predict(X)[0]

    return pred


HTML_PAGE = """
<!DOCTYPE html>
<html>
<head>
    <title>Sentiment Predictor</title>
</head>
<body>

    <h2>Sentiment Analysis (BoW + Linear SVC)</h2>

    <form method="POST">
        <textarea name="text">{{text}}</textarea><br><br>
        <button type="submit">Predict</button>
    </form>

    {% if result is not none %}
        <hr>
        <h3>Sentiment: {{sentiment}}</h3>
        <p><b>Raw Output:</b> {{result}}</p>
    {% endif %}

</body>
</html>
"""


@app.route("/", methods=["GET", "POST"])
def home():
    result = None
    sentiment = None
    text = ""

    if request.method == "POST":
        text = request.form["text"]
        result = predict_text(text)
        sentiment = LABELS.get(result, "unknown")

    return render_template_string(
        HTML_PAGE,
        result=result,
        sentiment=sentiment,
        text=text
    )


@app.route("/predict_api", methods=["POST"])
def predict_api():
    data = request.get_json()

    text = data.get("text", "")

    result = predict_text(text)

    return jsonify({
        "result": str(result),
        "sentiment": LABELS.get(result, "unknown")
    })


if __name__ == "__main__":
    app.run(debug=True)