from flask import Flask, render_template, request, jsonify
from nlp import preprocess_text, classify_text, generate_reply
from utils import read_file_text
from dotenv import load_dotenv
import json
import os
import joblib
import logging
import gc
import sys

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Garantir que o modelo existe ao iniciar
def initialize_model():
    try:
        if not os.path.exists("models/tfidf_lr.joblib"):
            logger.info("[TRAIN] Modelo nao encontrado. Treinando...")
            from train_model import pipeline, TEXTS, LABELS
            os.makedirs("models", exist_ok=True)
            pipeline.fit(TEXTS, LABELS)
            joblib.dump(pipeline, "models/tfidf_lr.joblib")
            logger.info("[OK] Modelo treinado e salvo com sucesso!")
        else:
            logger.info("[OK] Modelo carregado do arquivo existente")
    except Exception as e:
        logger.error(f"[ERRO] Erro ao inicializar modelo: {e}")

initialize_model()
load_dotenv()

app = Flask(__name__)

def limpar_cache():
    """Limpa cache e memoria para nao contaminar proximas requisicoes"""
    try:
        gc.collect()
        logger.info("[CACHE] Limpeza de cache realizada")
    except Exception as e:
        logger.error(f"[AVISO] Erro ao limpar cache: {e}")

@app.before_request
def before_request():
    """Limpa cache antes de cada requisicao"""
    if request.method == "POST":
        logger.info("[REQ] Limpando cache antigo...")
        limpar_cache()

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

@app.route("/process", methods=["POST"])
def process():
    try:
        text = request.form.get("text", "").strip()
        uploaded = request.files.get("file")

        # Ler texto do arquivo se fornecido
        if uploaded and uploaded.filename != "":
            try:
                text = read_file_text(uploaded)
                logger.info(f"📄 Arquivo lido: {uploaded.filename}")
            except Exception as e:
                logger.error(f"[ERRO] Erro ao ler arquivo: {e}")
                return render_template("index.html", result={
                    "category": "Erro",
                    "score": 0,
                    "reply": f"Erro ao ler arquivo: {str(e)}"
                })

        # Validar se há texto
        if not text:
            return render_template("index.html", result={
                "category": "Nenhum texto fornecido",
                "score": 0,
                "reply": "Forneça um texto ou arquivo."
            })

        # Pré-processar texto
        processed = preprocess_text(text)
        logger.info(f"[OK] Texto pre-processado: {len(processed['clean_text'])} caracteres")

        # Classificar texto
        category, score = classify_text(processed["clean_text"])
        logger.info(f"[INFO] Classificacao: {category} (confianca: {score:.2f})")

        # Obter dados do usuário
        user_email = request.form.get("email", "").strip()
        user_nome = request.form.get("nome", "").strip()
        user_assunto = request.form.get("assunto", "").strip()

        # Gerar resposta
        reply_json = generate_reply(
            email_text=text,
            category=category,
            user_name=user_nome,
            user_email=user_email,
            user_subject=user_assunto
        )

        # Processar resposta JSON
        structured_reply = json.loads(reply_json)
        
        reply_text = structured_reply.get("reply", "Não foi possível gerar resposta.")
        reply_category = structured_reply.get("category", category)
        reply_score = structured_reply.get("score", score)
        
        # Formatar resposta para HTML
        reply_html = reply_text.replace("\r\n", "\n").replace("\n\n", "<p>").replace("\n", "<br>")
        
        result = {
            "category": reply_category,
            "score": f"{reply_score:.2f}",
            "reply": reply_html
        }
        
        logger.info(f"[OK] Resposta gerada com sucesso")
        limpar_cache()  # Limpar cache apos processar
        return render_template("index.html", result=result)
        
    except json.JSONDecodeError as e:
        logger.error(f"[ERRO] Erro ao decodificar JSON: {e}")
        limpar_cache()  # Limpar cache mesmo em erro
        return render_template("index.html", result={
            "category": "Erro",
            "score": 0,
            "reply": f"Erro ao processar resposta: {str(e)}"
        })
    except Exception as e:
        logger.error(f"[ERRO CRITICO] {str(e)}", exc_info=True)
        limpar_cache()  # Limpar cache mesmo em erro
        return render_template("index.html", result={
            "category": "Erro",
            "score": 0,
            "reply": f"Erro ao processar: {str(e)}"
        }), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)