import re
import os
import json
import joblib
import logging
from typing import Tuple, Dict
from nltk.stem import RSLPStemmer
from sklearn.feature_extraction.text import TfidfVectorizer

logger = logging.getLogger(__name__)

# Stopwords
try:
    from nltk.corpus import stopwords
    _stop_words = set(stopwords.words("portuguese"))
except Exception:
    _stop_words = set([
        "de","da","do","em","um","uma","para","com","é","e","o","a","os","as",
        "no","na","por","se","que","ao","à","este","esse","aquele"
    ])

# Stemmer
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
        return {"original_text": "", "clean_text": "", "token_count": 0, "word_count": 0}
    
    text = text.lower().strip()
    words = text.split()
    
    filtered_words = [
        stemmer.stem(word) for word in words 
        if word not in _stop_words and len(word) > 2
    ]
    
    clean_text = " ".join(filtered_words)
    return {
        "original_text": text,
        "clean_text": clean_text,
        "token_count": len(filtered_words),
        "word_count": len(words)
    }

def classify_text(clean_text: str) -> Tuple[str, float]:
    """Classifica o texto usando modelo treinado ou heurística."""
    if not clean_text or len(clean_text.strip()) == 0:
        return "Improdutivo", 0.0
    
    words = clean_text.split()
    word_count = len(words)
    
    # TEXTO MUITO CURTO = SEMPRE IMPRODUTIVO
    if word_count < 3:
        logger.warning(f"[AVISO] Texto muito curto ({word_count} palavras) - classificado como Improdutivo")
        return "Improdutivo", 0.1
    
    # Limite de confiança baseado no tamanho
    if word_count < 50:
        max_confidence = 0.60
    elif word_count < 100:
        max_confidence = 0.70
    elif word_count < 200:
        max_confidence = 0.85
    else:
        max_confidence = 1.0
    
    # Se modelo disponível
    if pipeline is not None:
        try:
            prediction = pipeline.predict([clean_text])[0]
            probabilities = pipeline.predict_proba([clean_text])[0]
            category = "Produtivo" if prediction == 1 else "Improdutivo"
            confidence = min(float(max(probabilities)), max_confidence)
            logger.info(f"[MODELO] {word_count} palavras -> {category} ({confidence:.2f})")
            return category, confidence
        except Exception as e:
            logger.error(f"[ERRO] Erro ao usar modelo: {e}")
    
    # Heurística se não houver modelo
    productive_keywords = {
        "urgente", "suporte", "erro", "bug", "problema", "solicitação",
        "informação", "status", "dúvida", "questão", "reunião", "aprovação",
        "acesso", "integração", "implementação", "feedback", "relatório",
        "correção", "alteração", "backup", "dados", "projeto", "prazo"
    }
    
    productive_count = sum(1 for word in words if word in productive_keywords)
    score = 0.0
    category = "Improdutivo"
    
    if word_count < 30:
        score = 0.2
        category = "Improdutivo"
    elif 30 <= word_count < 50:
        if productive_count >= 2:
            score = 0.5
            category = "Produtivo"
        else:
            score = 0.3
            category = "Improdutivo"
    elif 50 <= word_count < 100:
        score = 0.4 if productive_count == 0 else 0.6
        category = "Improdutivo" if productive_count == 0 else "Produtivo"
    elif 100 <= word_count < 200:
        score = 0.6 if productive_count == 0 else 0.75
        category = "Improdutivo" if productive_count == 0 else "Produtivo"
    else:
        score = 0.7 if productive_count == 0 else 0.9
        category = "Produtivo"
    
    logger.info(f"[HEURISTICA] {word_count} palavras -> {category} ({score:.2f})")
    return category, score

def calculate_subject_confidence_boost(user_subject: str, user_name: str = "", user_email: str = "") -> float:
    """
    Calcula um multiplicador de confiança baseado no ASSUNTO e dados do usuário.
    """
    boost = 1.0
    
    print("\n" + "="*80)
    print("[ANALISE DE ASSUNTO E DADOS]")
    print("="*80)
    print(f"Assunto: '{user_subject}'")
    print(f"Nome: '{user_name}'")
    print(f"Email: '{user_email}'")
    
    # Verificar qualidade do assunto
    if not user_subject or user_subject.strip() == "":
        print("  [RESULTADO] Sem assunto detectado - REDUZINDO 0.05")
        boost -= 0.05
    elif len(user_subject) < 5:
        print(f"  [RESULTADO] Assunto muito curto ({len(user_subject)} chars) - REDUZINDO 0.10")
        boost -= 0.1
    elif len(user_subject) > 100:
        print(f"  [RESULTADO] Assunto muito longo ({len(user_subject)} chars) - REDUZINDO 0.05")
        boost -= 0.05
    else:
        # Assunto profissional/legítimo
        professional_words = {
            "urgente", "suporte", "erro", "problema", "solicitação",
            "informação", "status", "dúvida", "reunião", "aprovação",
            "projeto", "prazo", "contato", "feedback", "relatório"
        }
        subject_lower = user_subject.lower()
        if any(word in subject_lower for word in professional_words):
            print(f"  [RESULTADO] Assunto PROFISSIONAL detectado - AUMENTANDO 0.15")
            boost += 0.15
        else:
            print(f"  [RESULTADO] Assunto genérico/normal")
    
    # Verificar qualidade do nome
    if user_name and len(user_name) > 3:
        print(f"  [RESULTADO] Nome completo '{user_name}' - AUMENTANDO 0.10")
        boost += 0.1
    else:
        print(f"  [RESULTADO] Nome ausente ou muito curto")
    
    # Verificar email
    if user_email:
        if "@" in user_email and "." in user_email:
            if "gmail.com" in user_email or "hotmail.com" in user_email or "yahoo.com" in user_email:
                print(f"  [RESULTADO] Email pessoal (gmail/hotmail/yahoo) - AUMENTANDO 0.05")
                boost += 0.05
            elif any(domain in user_email for domain in ["empresa", "company", "corp", "inc"]):
                print(f"  [RESULTADO] Email CORPORATIVO - AUMENTANDO 0.20")
                boost += 0.2
            else:
                print(f"  [RESULTADO] Email padrão")
        else:
            print(f"  [RESULTADO] Email SUSPEITO - REDUZINDO 0.15")
            boost -= 0.15
    else:
        print(f"  [RESULTADO] Email não fornecido")
    
    boost = max(0.5, min(boost, 1.5))
    
    print(f"\n[MULTIPLICADOR ASSUNTO/EMAIL/NOME] = {boost:.2f} (0.5-1.5)")
    print("="*80 + "\n")
    
    return boost


def calculate_content_confidence_boost(email_text: str, word_count: int) -> float:
    """
    Calcula um multiplicador de confiança baseado no CONTEÚDO do email.
    """
    boost = 1.0
    
    print("\n" + "="*80)
    print("[ANALISE DE CONTEÚDO DO EMAIL]")
    print("="*80)
    print(f"Tamanho do texto: {len(email_text)} caracteres")
    print(f"Número de palavras: {word_count} palavras")
    
    if word_count < 20:
        print(f"  [RESULTADO] Texto MUITO CURTO ({word_count} palavras) - REDUZINDO 0.15")
        boost -= 0.15
    elif word_count > 500:
        print(f"  [RESULTADO] Texto MUITO LONGO ({word_count} palavras) - REDUZINDO 0.05")
        boost -= 0.05
    else:
        print(f"  [RESULTADO] Comprimento normal ({word_count} palavras) - AUMENTANDO 0.05")
        boost += 0.05
    
    # Verificar CAPS LOCK excessivo
    caps_count = sum(1 for c in email_text if c.isupper())
    if word_count > 0:
        caps_ratio = caps_count / len(email_text)
        print(f"  Caracteres em MAIÚSCULA: {caps_count} ({caps_ratio:.1%})")
        if caps_ratio > 0.3:
            print(f"  [RESULTADO] Muitas MAIÚSCULAS ({caps_ratio:.1%}) - REDUZINDO 0.20 (SPAM!)")
            boost -= 0.2
    
    # Verificar pontuação
    question_count = email_text.count("?")
    exclamation_count = email_text.count("!")
    
    print(f"  Interrogações (?): {question_count}")
    print(f"  Exclamações (!): {exclamation_count}")
    
    if question_count > 2:
        print(f"  [RESULTADO] Muitas interrogações - parece dúvida genuína - AUMENTANDO 0.10")
        boost += 0.1
    
    if exclamation_count > 5:
        print(f"  [RESULTADO] Muitas exclamações - REDUZINDO 0.10 (suspeito)")
        boost -= 0.1
    
    # Verificar palavras-chave profissionais
    professional_keywords = {
        "urgente", "suporte", "erro", "bug", "problema", "solicitação",
        "informação", "status", "dúvida", "questão", "reunião", "aprovação",
        "acesso", "integração", "implementação", "feedback", "relatório",
        "correção", "alteração", "backup", "dados", "projeto", "prazo",
        "ajuda", "assistência", "necessário", "importante", "atenção"
    }
    
    email_lower = email_text.lower()
    prof_keywords_found = [word for word in professional_keywords if word in email_lower]
    prof_count = len(prof_keywords_found)
    
    print(f"  Palavras-chave profissionais encontradas: {prof_count}")
    if prof_keywords_found:
        print(f"    → {', '.join(prof_keywords_found[:5])}")
    
    if prof_count >= 3:
        print(f"  [RESULTADO] {prof_count} palavras profissionais - AUMENTANDO 0.15")
        boost += 0.15
    
    boost = max(0.5, min(boost, 1.5))
    
    print(f"\n[MULTIPLICADOR CONTEÚDO] = {boost:.2f} (0.5-1.5)")
    print("="*80 + "\n")
    
    return boost

# Gemini AI
try:
    from google import genai
    _has_genai = True
    logger.info("[OK] Biblioteca google-genai importada com sucesso")
except Exception as e:
    _has_genai = False
    logger.error(f"[ERRO] Falha ao importar google-genai: {e}")

_genai_api_key = os.getenv("GENAI_API_KEY")
genai_client = None

logger.info(f"[INFO] GENAI_API_KEY presente: {bool(_genai_api_key)}")
if _genai_api_key:
    logger.info(f"[INFO] GENAI_API_KEY tamanho: {len(_genai_api_key)} caracteres")
    logger.info(f"[INFO] GENAI_API_KEY primeiros 10 chars: {_genai_api_key[:10]}***")

if _has_genai and _genai_api_key:
    try:
        logger.info("[TENTANDO] Conectar ao Gemini AI...")
        genai_client = genai.Client(api_key=_genai_api_key)
        logger.info("[OK] Gemini AI cliente criado com sucesso")
        print(f"[OK] genai_client: {genai_client}")
    except Exception as e:
        logger.error(f"[ERRO] Falha ao inicializar Gemini: {e}")
        logger.error(f"[ERRO] Tipo de erro: {type(e).__name__}")
        genai_client = None
else:
    if not _has_genai:
        logger.warning("[AVISO] Biblioteca google-genai nao disponivel")
    if not _genai_api_key:
        logger.warning("[AVISO] GENAI_API_KEY nao configurada no .env")

def generate_reply(email_text: str, category: str, user_name: str = "", 
                   user_email: str = "", user_subject: str = "", classification_score: float = 0.5) -> str:
    """Gera resposta automática personalizada com score adaptado ao conteúdo.
    
    Args:
        email_text: Corpo do email original
        category: Categoria detectada (Produtivo/Improdutivo)
        user_name: Nome do remetente
        user_email: Email do remetente
        user_subject: Assunto do email
        classification_score: Score bruto da classificação
    """
    try:
        print("\n\n" + "█"*80)
        print("[GERACAO DE RESPOSTA - INICIADA]")
        print("█"*80)
        print(f"Categoria: {category}")
        print(f"Score BRUTO recebido: {classification_score:.2f}")
        
        # CALCULAR MULTIPLICADORES baseado em ASSUNTO e CONTEÚDO
        print("\n" + "-"*80)
        print("PASSO 1: Calculando BOOST DO ASSUNTO...")
        print("-"*80)
        subject_boost = calculate_subject_confidence_boost(user_subject, user_name, user_email)
        
        print("\n" + "-"*80)
        print("PASSO 2: Calculando BOOST DO CONTEÚDO...")
        print("-"*80)
        word_count = len(email_text.split())
        content_boost = calculate_content_confidence_boost(email_text, word_count)
        
        # Combinar os multiplicadores (média ponderada)
        print("\n" + "-"*80)
        print("PASSO 3: Combinando os multiplicadores...")
        print("-"*80)
        print(f"Subject Boost: {subject_boost:.2f} (pesa 40%)")
        print(f"Content Boost: {content_boost:.2f} (pesa 60%)")
        
        final_multiplier = (subject_boost * 0.4 + content_boost * 0.6)  # Conteúdo pesa mais
        print(f"Multiplicador final: ({subject_boost:.2f} × 0.4) + ({content_boost:.2f} × 0.6) = {final_multiplier:.2f}")
        
        # Aplicar o multiplicador ao score
        print("\n" + "-"*80)
        print("PASSO 4: Aplicando multiplicador ao score...")
        print("-"*80)
        print(f"Cálculo: {classification_score:.2f} × {final_multiplier:.2f} = {classification_score * final_multiplier:.2f}")
        
        adjusted_score = classification_score * final_multiplier
        adjusted_score = max(0.0, min(adjusted_score, 1.0))  # Garantir 0-1
        
        print(f"Score AJUSTADO (final): {adjusted_score:.2f}")
        print(f"Mudança: {'+' if adjusted_score > classification_score else ''}{(adjusted_score - classification_score):.2f} ({((adjusted_score - classification_score)/classification_score * 100):+.1f}%)")
        
        logger.info(f"[GERANDO] Resposta para: {category}")
        logger.info(f"[INFO] Score bruto: {classification_score:.2f}")
        logger.info(f"[SCORE] Subject: {subject_boost:.2f} | Content: {content_boost:.2f} | Final: {final_multiplier:.2f} | Adjusted: {adjusted_score:.2f}")
        
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
                logger.info("[TENTANDO] Chamar Gemini API...")
                response = genai_client.models.generate_content(
                    model="gemini-2.5-flash",
                    contents=prompt
                )
                reply_text = response.text.strip()
                logger.info("[OK] Resposta recebida do Gemini")
            except Exception as e:
                logger.error(f"[ERRO] Falha ao chamar Gemini: {e}")
                logger.error(f"[ERRO] Tipo: {type(e).__name__}")
                logger.warning("[FALLBACK] Usando template...")
                reply_text = _get_template_reply(category, user_name, user_subject, email_text)
        else:
            logger.warning("[AVISO] genai_client nao disponivel, usando template")
            reply_text = _get_template_reply(category, user_name, user_subject, email_text)
        
        # USAR O SCORE AJUSTADO, NAO O SCORE BRUTO!
        score = adjusted_score
        
        print("\n" + "█"*80)
        print(f"[RESULTADO FINAL]")
        print("█"*80)
        print(f"Categoria: {category}")
        print(f"Score Original: {classification_score:.2f}")
        print(f"Score Ajustado: {score:.2f}")
        print(f"Resposta: {reply_text[:80]}...")
        print("█"*80 + "\n")
        
        logger.info(f"[OK] Resposta pronta com score ajustado: {score:.2f}")
        return json.dumps({
            "category": category,
            "score": score,
            "reply": reply_text
        }, ensure_ascii=False)
    
    except Exception as e:
        logger.error(f"[ERRO] Erro ao gerar resposta: {e}")
        logger.error(f"[ERRO] Tipo: {type(e).__name__}")
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
