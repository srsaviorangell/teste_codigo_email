import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

def preprocess_text(text):
    """
    Função de pré-processamento NLP básica.
    Limpa, normaliza e tokeniza o texto.
    """

    text = text.lower()

    text = re.sub(r'[^a-záéíóúâêîôûãõàç\s]', '', text)

    tokens = word_tokenize(text, language='portuguese')

    stop_words = set(stopwords.words('portuguese'))
    filtered_tokens = [word for word in tokens if word not in stop_words and len(word) > 1]

    clean_text = ' '.join(filtered_tokens)

    return {
        "original": text,
        "tokens": filtered_tokens,
        "clean_text": clean_text
    }
