"""
Script de treinamento do modelo de classificação de emails.
Treina um modelo TF-IDF + Logistic Regression e salva em models/tfidf_lr.joblib
"""

import os
import joblib
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from nltk.stem import RSLPStemmer
from nltk.corpus import stopwords
from sample_data import TEXTS, LABELS

try:
    stop_words = set(stopwords.words("portuguese"))
except:
    stop_words = set([
        "de","da","do","em","um","uma","para","com","é","e","o","a","os","as",
        "no","na","por","se","que","ao","à","este","esse","aquele"
    ])

os.makedirs("models", exist_ok=True)

print("[INICIO] Iniciando treinamento do modelo...")
print(f"[INFO] Total de amostras: {len(TEXTS)}")
print(f"[OK] Produtivos: {sum(LABELS)}")
print(f"[INFO] Improdutivos: {len(LABELS) - sum(LABELS)}")

pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(
        max_features=500,
        stop_words=list(stop_words),
        ngram_range=(1, 2),
        min_df=1,
        max_df=0.8,
        lowercase=True,
        token_pattern=r'(?u)\b\w+\b'
    )),
    ('classifier', LogisticRegression(
        max_iter=200,
        random_state=42,
        solver='lbfgs'
    ))
])

print("\n[TRAIN] Treinando modelo...")
pipeline.fit(TEXTS, LABELS)

model_path = "models/tfidf_lr.joblib"
joblib.dump(pipeline, model_path)
print(f"[OK] Modelo salvo em: {model_path}")

print("\n[TEST] Testando modelo:")
test_emails = [
    "Preciso de suporte urgente. Sistema fora do ar.",
    "Obrigado pela ajuda!",
    "Qual é o status do projeto?",
    "Parabéns pelo ótimo trabalho!"
]

for email in test_emails:
    pred = pipeline.predict([email])[0]
    proba = pipeline.predict_proba([email])[0]
    category = "Produtivo" if pred == 1 else "Improdutivo"
    confidence = max(proba) * 100
    print(f"  [EMAIL] '{email[:40]}...' -> {category} ({confidence:.1f}%)")

print("\n[FIM] Treinamento concluido!")