import re
import os
import json
import joblib
from typing import Tuple, Dict
from nltk.stem import RSLPStemmer
from sklearn.feature_extraction.text import TfidfVectorizer

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
vectorizer = None
model = None

def load_model():
    global vectorizer, model
    if os.path.exists(MODEL_PATH):
        try:
            pipeline = joblib.load(MODEL_PATH)
            print(f"✅ Modelo carregado de: {MODEL_PATH}")
            return pipeline, pipeline
        except Exception as e:
            print(f"⚠️ Erro ao carregar modelo: {e}")
            return None, None
    else:
        print(f"⚠️ Modelo não encontrado em {MODEL_PATH}")
        print("💡 Execute: python train_model.py")
        return None, None

pipeline, _ = load_model()

def preprocess_text(text: str) -> Dict:
    """Pré-processa o texto removendo stopwords e aplicando stemming."""
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
    """
    if not clean_text or len(clean_text.strip()) == 0:
        return "Improdutivo", 0.0
    
    if pipeline is not None:
        try:
            prediction = pipeline.predict([clean_text])[0]
            probabilities = pipeline.predict_proba([clean_text])[0]
            
            category = "Produtivo" if prediction == 1 else "Improdutivo"
            confidence = max(probabilities)
            
            return category, confidence
        except Exception as e:
            print(f"⚠️ Erro ao usar modelo: {e}")
    
    print("⚠️ Usando heurística (modelo não disponível)")
    
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
        print("✅ Gemini AI habilitado")
    except Exception as e:
        print(f"⚠️ Erro ao inicializar Gemini: {e}")

def generate_reply(email_text: str, category: str, user_name: str = "", 
                   user_email: str = "", user_subject: str = "") -> str:
    """
    Gera resposta automática personalizada usando Gemini (se disponível) ou template.
    Retorna string JSON com {category, score, reply}.
    
    Args:
        email_text: Corpo do email original
        category: Categoria detectada (Produtivo/Improdutivo)
        user_name: Nome do remetente
        user_email: Email do remetente
        user_subject: Assunto do email
    """
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
        except Exception as e:
            print(f"⚠️ Erro ao chamar Gemini: {e}")
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