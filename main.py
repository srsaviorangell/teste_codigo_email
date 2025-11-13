from flask import Flask, render_template, request
from nlp import preprocess_text, classify_text, generate_reply
from utils import read_file_text
from dotenv import load_dotenv
import re
import os
import joblib

if not os.path.exists("models/tfidf_lr.joblib"):
    from train_model import pipeline, TEXTS, LABELS
    os.makedirs("models", exist_ok=True)
    pipeline.fit(TEXTS, LABELS)
    joblib.dump(pipeline, "models/tfidf_lr.joblib")

load_dotenv()

app = Flask(__name__)

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

@app.route("/process", methods=["POST"])
def process():
    text = request.form.get("text", "").strip()
    uploaded = request.files.get("file")

    if uploaded and uploaded.filename != "":
        text = read_file_text(uploaded)
        print("=== TEXTO LIDO DO ARQUIVO ===")
        print(text)
        print("==============================")

    if not text:
        return render_template("index.html", result={
            "category": "Nenhum texto fornecido",
            "score": 0,
            "reply": "Forneça um texto ou arquivo."
        })

    processed = preprocess_text(text)
    print("=== TEXTO PRÉ-PROCESSADO ===")
    print(processed["clean_text"])
    print("=============================")

    category, score = classify_text(processed["clean_text"])
    user_email = request.form.get("email", "").strip()
    user_nome = request.form.get("nome", "").strip()
    user_assunto = request.form.get("assunto", "").strip()

    reply_json = generate_reply(
        email_text=text,
        category=category,
        user_name=user_nome,
        user_email=user_email,
        user_subject=user_assunto
    )

    import json
    try:
        structured_reply = json.loads(reply_json)
        
        reply_text = structured_reply.get("reply", "Não foi possível gerar resposta.")
        reply_category = structured_reply.get("category", category)
        reply_score = structured_reply.get("score", score)
        
        reply_html = reply_text.replace("\r\n", "\n").replace("\n\n", "<p>").replace("\n", "<br>")
        
        result = {
            "category": reply_category,
            "score": f"{reply_score:.2f}",
            "reply": reply_html
        }
    except json.JSONDecodeError as e:
        print(f"⚠️ Erro ao decodificar JSON: {e}")
        reply_html = str(reply_json).replace("\r\n", "\n").replace("\n\n", "<p>").replace("\n", "<br>")
        result = {
            "category": category,
            "score": f"{score:.2f}",
            "reply": reply_html
        }

    return render_template("index.html", result=result)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)