import re
import os
import json
import joblib
import logging
from typing import Tuple, Dict
from nltk.stem import RSLPStemmer
from sklearn.feature_extraction.text import TfidfVectorizer

logger = logging.getLogger(__name__)

try:
    from nltk.corpus import stopwords
    _stop_words = set(stopwords.words("portuguese"))
except Exception:
    _stop_words = set([
        "de","da","do","em","um","uma","para","com","é","e","o","a","os","as",
        "no","na","por","se","que","ao","à","este","esse","aquele"
    ])

try:
    stemmer = RSLPStemmer()
except Exception:
    class _DummyStemmer:
        def stem(self, word):
            return word
    stemmer = _DummyStemmer()

MODEL_PATH = "models/tfidf_lr.joblib"
pipeline = None

def load_model():
    global pipeline
    if os.path.exists(MODEL_PATH):
        try:
            pipeline = joblib.load(MODEL_PATH)
            logger.info(f"[OK] Modelo carregado de: {MODEL_PATH}")
            return pipeline
        except Exception as e:
            logger.error(f"[ERRO] Erro ao carregar modelo: {e}")
            return None
    else:
        logger.warning(f"[AVISO] Modelo nao encontrado em {MODEL_PATH}")
        logger.info("[INFO] Execute: python train_model.py")
        return None

pipeline = load_model()

def preprocess_text(text: str) -> Dict:
    """Pré-processa o texto removendo stopwords e aplicando stemming."""
    if not text:
        return {
            "original_text": "",
            "clean_text": "",
            "token_count": 0,
            "word_count": 0
        }
    
    text = text.lower().strip()
    words = text.split()
    
    filtered_words = [
        stemmer.stem(word) for word in words 
        if word not in _stop_words and len(word) > 2
    ]
    
    clean_text = " ".join(filtered_words)
    token_count = len(filtered_words)
    
    return {
        "original_text": text,
        "clean_text": clean_text,
        "token_count": token_count,
        "word_count": len(words)
    }

def classify_text(clean_text: str) -> Tuple[str, float]:
    """
    Classifica o texto usando modelo treinado (TF-IDF + LR).
    Se modelo não existir, usa heurística baseada em comprimento e contexto.
    
    CALCULO DE CONFIANCA:
    - Se modelo disponivel: usa probabilidade do modelo (0.0-1.0)
    - Se sem modelo: usa heuristica baseada em tamanho + palavras-chave
      * Texto < 50 palavras + sem keywords = 0.20
      * Texto < 50 palavras + com keywords = 0.30
      * Texto 50-100 palavras + sem keywords = 0.31
      * Texto 50-100 palavras + com keywords = 0.50
      * Texto 100-200 palavras + sem keywords = 0.51
      * Texto 100-200 palavras + com keywords = 0.65
      * Texto > 200 palavras + sem keywords = 0.70
      * Texto > 200 palavras + com keywords = 0.90
    """
    if not clean_text or len(clean_text.strip()) == 0:
        return "Improdutivo", 0.0
    
    if pipeline is not None:
        try:
            prediction = pipeline.predict([clean_text])[0]
            probabilities = pipeline.predict_proba([clean_text])[0]
            
            category = "Produtivo" if prediction == 1 else "Improdutivo"
            confidence = float(max(probabilities))
            
            logger.info(f"[OK] Classificacao com modelo: {category} ({confidence:.2f})")
            return category, confidence
        except Exception as e:
            logger.error(f"[ERRO] Erro ao usar modelo: {e}")
    
    logger.warning("[AVISO] Usando heuristica (modelo nao disponivel)")
    
    words = clean_text.split()
    length = len(words)
    
    productive_keywords = {
        "urgente", "suporte", "erro", "bug", "problema", "solicitação",
        "informação", "status", "dúvida", "questão", "reunião", "aprovação",
        "acesso", "integração", "implementação", "feedback", "relatório",
        "correção", "alteração", "backup", "dados", "projeto", "prazo"
    }
    
    productive_count = sum(1 for word in words if word in productive_keywords)
    contexto = productive_count > 0
    
    score = 0.0
    category = "Improdutivo"
    
    if length < 50:
        score = 0.2 if not contexto else 0.3
        category = "Improdutivo" if not contexto else "Produtivo"
    elif 50 <= length <= 100:
        score = 0.31 if not contexto else 0.5
        category = "Improdutivo" if not contexto else "Produtivo"
    elif 100 < length <= 200:
        score = 0.51 if not contexto else 0.65
        category = "Improdutivo" if not contexto else "Produtivo"
    else:  
        score = 0.7 if not contexto else 0.90
        category = "Produtivo"
    
    return category, score

try:
    from google import genai
    _has_genai = True
except Exception:
    _has_genai = False

_genai_api_key = os.getenv("GENAI_API_KEY")
genai_client = None

if _has_genai and _genai_api_key:
    try:
        genai_client = genai.Client(api_key=_genai_api_key)
        logger.info("[OK] Gemini AI habilitado")
    except Exception as e:
        logger.warning(f"[AVISO] Erro ao inicializar Gemini: {e}")

def generate_reply(email_text: str, category: str, user_name: str = "", 
                   user_email: str = "", user_subject: str = "") -> str:
    """
    Gera resposta automática personalizada usando Gemini (se disponível) ou template.
    Retorna string JSON com {category, score, reply}.
    """
    try:
        prompt = f"""Você é um assistente de suporte corporativo profissional e atencioso.

Dados do email recebido:
- Remetente: {user_name if user_name else "Não informado"}
- Email: {user_email if user_email else "Não informado"}
- Assunto: {user_subject if user_subject else "Sem assunto"}
- Corpo: {email_text[:300]}

Categoria detectada: {category}

Gere uma resposta profissional, personalizada e concisa (máximo 6 linhas) que:
1. Cumprimente o remetente pelo nome (se fornecido)
2. Reconheça o assunto/conteúdo
3. Forneça orientação apropriada
4. Ofereça disponibilidade

Não adicione explicações, apenas a resposta."""
        
        if genai_client:
            try:
                response = genai_client.models.generate_content(
                    model="gemini-2.5-flash",
                    contents=prompt
                )
                reply_text = response.text.strip()
                logger.info("[OK] Resposta gerada com Gemini")
            except Exception as e:
                logger.warning(f"[AVISO] Erro ao chamar Gemini: {e}")
                reply_text = _get_template_reply(category, user_name, user_subject, email_text)
        else:
            reply_text = _get_template_reply(category, user_name, user_subject, email_text)
        
        score = 0.85 if category == "Produtivo" else 0.65
        
        result = {
            "category": category,
            "score": score,
            "reply": reply_text
        }
        
        return json.dumps(result, ensure_ascii=False)
    
    except Exception as e:
        logger.error(f"[ERRO] Erro ao gerar resposta: {e}")
        return json.dumps({
            "category": "Erro",
            "score": 0.0,
            "reply": "Erro ao gerar resposta automatica."
        }, ensure_ascii=False)

def _get_template_reply(category: str, user_name: str = "", user_subject: str = "", 
                        email_text: str = "") -> str:
    """Retorna resposta templada personalizada baseada na categoria e dados do usuário."""
    
    greeting = f"Prezado(a) {user_name}" if user_name else "Prezado(a)"
    
    content_preview = email_text[:50] if email_text else "(sem conteúdo)"
    word_count = len(email_text.split()) if email_text else 0
    
    if word_count < 30:
        content_char = "é bastante breve"
    elif word_count < 80:
        content_char = "é concisa"
    elif word_count < 200:
        content_char = "é detalhada"
    else:
        content_char = "é muito completa"
    
    subject_text = f"'{user_subject}'" if user_subject else "(sem assunto)"
    
    if category == "Produtivo":
        return f"""{greeting}, Agradecemos o seu contato. Recebemos seu e-mail com o assunto {subject_text}, porém o conteúdo da mensagem ('{content_preview}') {content_char}. Para que possamos dar o devido encaminhamento, poderia nos fornecer mais informações? Permanecemos à disposição."""
    else:
        return f"""{greeting}, Agradecemos o seu contato sobre {subject_text}. Recebemos sua mensagem ('{content_preview}') e valorizamos sua consideração. Sua contribuição é importante para nós. Muito obrigado e continue contando conosco."""