# 📧 Classificador de Emails com IA

Uma aplicação web moderna que utiliza **inteligência artificial** para classificar emails em categorias (Produtivo/Improdutivo) e gerar respostas automáticas sugeridas.

## ✨ Funcionalidades

- ✅ **Upload de arquivos**: Suporta `.txt` e `.pdf`
- ✅ **Colar texto direto**: Cole o corpo do email no formulário
- ✅ **Classificação inteligente**: Produtivo ou Improdutivo
- ✅ **Scoring dinâmico**: Baseado em comprimento e contexto
- ✅ **Geração de resposta**: Automática via IA (Gemini ou fallback local)
- ✅ **Interface moderna**: Card visual, badge colorida, botão copiar
- ✅ **Offline-ready**: Funciona mesmo sem API externa (fallback templated)

---

## 🎯 Critério de Avaliação (Scoring)

O sistema avalia cada email usando 3 critérios:

### 1. **Comprimento do Texto**
- `< 50 palavras`: Texto muito curto
- `50-100 palavras`: Texto curto/médio
- `100-200 palavras`: Texto médio
- `> 200 palavras`: Texto longo

### 2. **Detecção de Contexto**
**Palavras-chave Produtivas**: solicita, urgente, problema, erro, precisa, status, anexo, documento, por favor, retorno, ajuda, suporte, informação, dados, relatório, projeto, cliente, reunião, deadline, tarefa, resultado, feedback, crítico

**Palavras-chave Improdutivas**: feliz natal, obrigado, parabéns, boas festas, ótimo trabalho, valeu, abraço, agradecido, cumprimento, pessoalmente

### 3. **Score Normalizado (0.0 a 1.0)**

| Comprimento | Com Contexto | Sem Contexto |
|---|---|---|
| < 50 | 0.4 (Produtivo) | 0.2 (Improdutivo) |
| 50-100 | 0.4 (Produtivo) | 0.3 (Improdutivo) |
| 100-200 | 0.6 (Produtivo) | 0.5 (Improdutivo) |
| > 200 | 0.9 (Produtivo) | 0.7 (Improdutivo) |

---

## 🚀 Quick Start

### Requisitos
- Python 3.8+
- pip (gerenciador de pacotes)

### 1. Clonar/Baixar o Projeto
```bash
cd "teste_codigo_email"
```

### 2. Criar Virtual Environment
```powershell
python -m venv .venv
.\\.venv\\Scripts\\Activate.ps1
```

### 3. Instalar Dependências
```powershell
pip install -r requirements.txt
```

### 4. (Opcional) Baixar Recursos NLTK
```powershell
python -c "import nltk; nltk.download('rslp'); nltk.download('stopwords')"
```

### 5. Iniciar o Servidor
```powershell
python main.py
```

### 6. Acessar a Aplicação
Abra seu navegador em:
- **Local**: http://127.0.0.1:5000
- **Rede**: http://192.168.0.101:5000 (ou seu IP local)

---

## 🤖 Usando com IA (Gemini API)

Para ativar respostas via **Google Gemini** (em vez do fallback local):

### 1. Obter API Key
- Acesse: https://ai.google.dev/
- Crie um projeto e gere uma API Key
- Copie a chave

### 2. Configurar Variável de Ambiente

**Opção A: Arquivo `.env`** (recomendado)
```
GENAI_API_KEY=sua_chave_aqui
```

**Opção B: Variável do Sistema (Windows)**
```powershell
[Environment]::SetEnvironmentVariable("GENAI_API_KEY", "sua_chave_aqui", "User")
```

### 3. Instalar SDK do Gemini
```powershell
pip install google-genai
```

### 4. Reiniciar o Servidor
```powershell
python main.py
```

Agora as respostas serão geradas via Gemini 2.5 Flash! 🚀

---

## 📁 Estrutura do Projeto

```
teste de codigo outra vaga/
├── main.py                 # Backend Flask
├── nlp.py                  # Lógica de classificação e IA
├── utils.py                # Utilitários (leitura de arquivos)
├── preprocess.py           # (Opcional) Pré-processamento adicional
├── requirements.txt        # Dependências Python
├── templates/
│   └── index.html          # Interface web
├── static/
│   └── style.css           # Estilos CSS
├── .env                    # Variáveis de ambiente (não versionado)
└── README.md               # Este arquivo
```

---

## 🔧 Estrutura de Arquivos Principais

### `main.py` - Backend Flask
- Rota `/` : Página inicial
- Rota `/process` (POST) : Processa email e retorna resultado
- Extração robusta de resposta da IA

### `nlp.py` - Processamento NLP
- `preprocess_text()` : Limpeza e normalização de texto
- `classify_text()` : Classificação com scoring inteligente
- `generate_reply()` : Geração de resposta (Gemini ou fallback)

### `utils.py` - Utilitários
- `read_file_text()` : Leitura de `.txt` e `.pdf`

### `templates/index.html` - Interface
- Formulário para upload/cola de texto
- Card de resultado com badge, score e botão copiar
- JavaScript para calcular barra de progresso

### `static/style.css` - Estilos
- Design moderno com gradiente
- Animações suaves
- Responsivo (mobile-friendly)

---

## 🎨 Interface

### Formulário
- Nome completo (opcional)
- Assunto do email (opcional)
- Email do remetente (opcional)
- Textarea para colar texto
- Input para upload de arquivo

### Resultado
- **Header**: Título + Badge colorida (verde=Produtivo, vermelho=Improdutivo)
- **Métricas**: Categoria e barra de confiança (0-100%)
- **Resposta**: Card com texto em destaque
- **Botão Copiar**: Copia a resposta com feedback visual (✅ Copiado!)

---

## 📊 Exemplos de Uso

### Exemplo 1: Email Produtivo Longo
```
Prezado(a),

Enviamos em anexo o relatório completo do projeto. 
Conforme solicitado, incluímos os dados de performance e feedback dos clientes.
Por favor, revise e nos retorne com suas considerações até a próxima reunião.
Qualquer dúvida, estou à disposição.

Atenciosamente,
João Silva
```
**Resultado**: Produtivo | Score: 0.90

### Exemplo 2: Email Improdutivo Curto
```
Oi! Feliz Natal para você! 
Aproveite as festas! 🎄
```
**Resultado**: Improdutivo | Score: 0.20

### Exemplo 3: Email Produtivo Curto
```
Urgente! Temos um problema crítico no sistema de produção.
Preciso de ajuda imediata!
```
**Resultado**: Produtivo | Score: 0.40

---

## 🔐 Segurança

- **Variáveis de ambiente**: API keys não são commitadas no código
- **Sanitização**: Inputs são processados de forma segura
- **HTML Escape**: Responses são escapadas, exceto onde necessário (quebras formatadas)

---

## 🐛 Troubleshooting

### Erro: "Resource rslp not found"
- Solução: Instale manualmente `python -c "import nltk; nltk.download('rslp')"`

### Erro: "genai.Client() failed"
- Solução: Verifique se `GENAI_API_KEY` está configurada
- Fallback: O app usa resposta templated local sem erro

### Erro: "Não consegue acessar http://127.0.0.1:5000"
- Solução: Verifique se Flask iniciou sem erros
- Verifique porta 5000: `netstat -ano | findstr :5000`

### Resposta muito curta/estranha
- Solução: Verifique o comprimento do email (mínimo 50 palavras para melhor resultado)

---

## 📚 Tecnologias Usadas

- **Backend**: Flask 3.1.2
- **NLP**: NLTK, scikit-learn, TF-IDF
- **AI**: Google Generative AI (Gemini 2.5 Flash)
- **PDF**: pdfminer.six
- **Frontend**: HTML5, CSS3, Vanilla JavaScript
- **Environment**: Python-dotenv

---

## 📝 Licença

Este projeto é fornecido como está, sem garantias. Use livremente para fins educacionais e comerciais.

---

## 🤝 Suporte

Caso tenha dúvidas ou problemas:
1. Verifique o README acima
2. Revise os logs do servidor (mensagens de erro no terminal)
3. Teste com exemplos simples primeiro
4. Verifique as variáveis de ambiente

---

**Desenvolvido com ❤️ para o desafio técnico de classificação de emails**
