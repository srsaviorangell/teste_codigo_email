import os
import tempfile
from pdfminer.high_level import extract_text

def read_file_text(uploaded_file):
    """Lê o conteúdo de um arquivo .txt ou .pdf e retorna o texto."""
    filename = uploaded_file.filename.lower()

    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(filename)[1]) as tmp:
        uploaded_file.save(tmp.name)
        temp_path = tmp.name

    try:
        if filename.endswith(".txt"):
            with open(temp_path, "r", encoding="utf-8", errors="ignore") as f:
                text = f.read()
        elif filename.endswith(".pdf"):
            text = extract_text(temp_path)
        else:
            text = ""
    finally:
        os.remove(temp_path) 

    return text
