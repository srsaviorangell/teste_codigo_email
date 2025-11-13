import unittest
import json
import sys
from io import StringIO
from unittest.mock import patch, MagicMock
from main import app
from nlp import (
    preprocess_text, 
    classify_text, 
    generate_reply,
    _get_template_reply
)


class TestPreprocessing(unittest.TestCase):
    """Testes para pré-processamento de texto."""
    
    def test_preprocess_basic(self):
        """Testa pré-processamento básico."""
        result = preprocess_text("Olá, como você está?")
        self.assertIn("clean_text", result)
        self.assertIn("token_count", result)
        self.assertGreater(result["token_count"], 0)
        self.assertIn("word_count", result)
        
    def test_preprocess_empty(self):
        """Testa pré-processamento com texto vazio."""
        result = preprocess_text("")
        self.assertEqual(result["token_count"], 0)
        
    def test_preprocess_removes_urls(self):
        """Testa que a função processa texto com URLs."""
        result = preprocess_text("Visite http://example.com para mais info")
        self.assertIn("clean_text", result)
        self.assertGreater(len(result["clean_text"]), 0)
        
    def test_preprocess_lowercase(self):
        """Testa conversão para minúsculas."""
        result = preprocess_text("TESTE COM MAIÚSCULAS")
        self.assertTrue(result["clean_text"].islower())


class TestClassification(unittest.TestCase):
    """Testes para classificação de emails."""
    
    def test_classify_productive(self):
        """Testa classificação de email produtivo."""
        text_productive = "Preciso de suporte urgente com erro no sistema"
        processed = preprocess_text(text_productive)
        category, score = classify_text(processed["clean_text"])
        self.assertEqual(category, "Produtivo")
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)
        
    def test_classify_unproductive(self):
        """Testa classificação de email improdutivo."""
        text_unproductive = "Parabéns pelo excelente trabalho, muito obrigado!"
        processed = preprocess_text(text_unproductive)
        category, score = classify_text(processed["clean_text"])
        self.assertEqual(category, "Improdutivo")
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)
        
    def test_classify_returns_valid_score(self):
        """Testa se score é válido."""
        text = "Teste geral de classificação"
        processed = preprocess_text(text)
        category, score = classify_text(processed["clean_text"])
        self.assertIsInstance(score, (int, float))
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)


class TestTemplateReply(unittest.TestCase):
    """Testes para geração de respostas template."""
    
    def test_template_productive_with_name(self):
        """Testa template produtivo com nome personalizado."""
        reply = _get_template_reply(
            category="Produtivo",
            user_name="João Silva",
            user_subject="Suporte",
            email_text="Preciso de ajuda"
        )
        self.assertIn("João Silva", reply)
        self.assertIn("Suporte", reply)
        self.assertIn("Prezado(a)", reply)
        
    def test_template_unproductive_with_name(self):
        """Testa template improdutivo com nome personalizado."""
        reply = _get_template_reply(
            category="Improdutivo",
            user_name="Maria Santos",
            user_subject="Agradecimento",
            email_text="Obrigado pela ajuda"
        )
        self.assertIn("Maria Santos", reply)
        self.assertIn("Agradecimento", reply)
        
    def test_template_without_name(self):
        """Testa template sem nome personalizado."""
        reply = _get_template_reply(
            category="Produtivo",
            user_subject="Teste"
        )
        self.assertIn("Prezado(a)", reply)
        self.assertNotIn("None", reply)
        
    def test_template_content_length_characterization(self):
        """Testa caracterização do comprimento do conteúdo."""
        reply_brief = _get_template_reply(
            category="Produtivo",
            email_text="OK"
        )
        self.assertIn("breve", reply_brief)
        
        reply_detailed = _get_template_reply(
            category="Produtivo",
            email_text=" ".join(["palavra"] * 150)
        )
        self.assertIn("detalhada", reply_detailed)


class TestGenerateReply(unittest.TestCase):
    """Testes para geração de respostas com Gemini fallback."""
    
    def test_generate_reply_returns_json(self):
        """Testa se generate_reply retorna JSON válido."""
        result_json = generate_reply(
            email_text="Teste de email",
            category="Produtivo",
            user_name="Teste User",
            user_email="teste@email.com",
            user_subject="Teste"
        )
        result = json.loads(result_json)
        self.assertIn("category", result)
        self.assertIn("score", result)
        self.assertIn("reply", result)
        
    def test_generate_reply_includes_category(self):
        """Testa se resposta inclui categoria correta."""
        result_json = generate_reply(
            email_text="Teste",
            category="Produtivo",
            user_name="Teste",
            user_email="teste@email.com",
            user_subject="Assunto"
        )
        result = json.loads(result_json)
        self.assertEqual(result["category"], "Produtivo")
        
    def test_generate_reply_has_valid_score(self):
        """Testa se score é válido."""
        result_json = generate_reply(
            email_text="Teste",
            category="Improdutivo",
            user_name="Teste",
            user_email="teste@email.com",
            user_subject="Assunto"
        )
        result = json.loads(result_json)
        score = result["score"]
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)
        
    def test_generate_reply_has_text(self):
        """Testa se resposta contém texto."""
        result_json = generate_reply(
            email_text="Teste",
            category="Produtivo",
            user_name="Teste",
            user_email="teste@email.com",
            user_subject="Assunto"
        )
        result = json.loads(result_json)
        self.assertGreater(len(result["reply"]), 0)


class TestFlaskRoutes(unittest.TestCase):
    """Testes para rotas Flask."""
    
    def setUp(self):
        """Configura cliente de teste."""
        self.app = app
        self.client = self.app.test_client()
        self.app.config['TESTING'] = True
        
    def test_index_get(self):
        """Testa rota GET /."""
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"email", response.data.lower() or b"form", response.data.lower())
        
    def test_process_empty_text(self):
        """Testa POST /process com texto vazio."""
        response = self.client.post("/process", data={
            "nome": "Teste",
            "assunto": "Teste",
            "email": "teste@email.com",
            "text": ""
        })
        self.assertEqual(response.status_code, 200)
        
    def test_process_with_text(self):
        """Testa POST /process com texto válido."""
        response = self.client.post("/process", data={
            "nome": "João Silva",
            "assunto": "Suporte Técnico",
            "email": "joao@email.com",
            "text": "Preciso de suporte urgente com erro no sistema"
        })
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Jo", response.data or b"o Silva", response.data)
        
    def test_process_with_all_fields(self):
        """Testa POST /process com todos os campos."""
        response = self.client.post("/process", data={
            "nome": "Maria Santos",
            "assunto": "Dúvida Geral",
            "email": "maria@email.com",
            "text": "Tenho uma dúvida sobre o produto. Como posso fazer?"
        })
        self.assertEqual(response.status_code, 200)


class TestIntegration(unittest.TestCase):
    """Testes de integração completa."""
    
    def test_full_pipeline_productive(self):
        """Testa pipeline completo para email produtivo."""
        email_text = "Existe um erro crítico no sistema de pagamento, preciso de suporte urgente"
        processed = preprocess_text(email_text)
        category, score = classify_text(processed["clean_text"])
        
        self.assertEqual(category, "Produtivo")
        self.assertGreaterEqual(score, 0.5)
        
        reply_json = generate_reply(
            email_text=email_text,
            category=category,
            user_name="Cliente VIP",
            user_email="vip@empresa.com",
            user_subject="Erro Crítico"
        )
        result = json.loads(reply_json)
        
        self.assertEqual(result["category"], "Produtivo")
        self.assertIn("Cliente VIP", result["reply"])
        self.assertIn("Erro Crítico", result["reply"])
        
    def test_full_pipeline_unproductive(self):
        """Testa pipeline completo para email improdutivo."""
        email_text = "Parabéns pelo excelente trabalho! Muito obrigado pela atenção!"
        processed = preprocess_text(email_text)
        category, score = classify_text(processed["clean_text"])
        
        self.assertEqual(category, "Improdutivo")
        
        reply_json = generate_reply(
            email_text=email_text,
            category=category,
            user_name="Cliente Satisfeito",
            user_email="satisfeito@empresa.com",
            user_subject="Agradecimento"
        )
        result = json.loads(reply_json)
        
        self.assertEqual(result["category"], "Improdutivo")
        self.assertIn("Cliente Satisfeito", result["reply"])


class TestEdgeCases(unittest.TestCase):
    """Testes de casos extremos."""
    
    def test_very_long_text(self):
        """Testa com texto muito longo."""
        long_text = " ".join(["palavra"] * 1000)
        processed = preprocess_text(long_text)
        self.assertGreater(processed["word_count"], 500)
        category, score = classify_text(processed["clean_text"])
        self.assertIn(category, ["Produtivo", "Improdutivo"])
        
    def test_special_characters(self):
        """Testa com caracteres especiais."""
        special_text = "Teste com @#$%&*()_+-=[]{}|;:',.<>?/~`"
        processed = preprocess_text(special_text)
        self.assertGreater(len(processed["clean_text"]), 0)
        
    def test_numbers_only(self):
        """Testa com apenas números."""
        numbers_text = "123 456 789"
        processed = preprocess_text(numbers_text)
        category, score = classify_text(processed["clean_text"])
        self.assertIn(category, ["Produtivo", "Improdutivo"])
        
    def test_unicode_characters(self):
        """Testa com caracteres Unicode."""
        unicode_text = "Teste com acentuação: àáâãäå èéêë ìíîï òóôõö ùúûü"
        processed = preprocess_text(unicode_text)
        self.assertGreater(len(processed["clean_text"]), 0)


def run_tests_with_report():
    """Executa testes com relatório detalhado."""
    print("\n" + "="*70)
    print("🧪 INICIANDO TESTES DA APLICAÇÃO DE CLASSIFICAÇÃO DE EMAILS")
    print("="*70 + "\n")
    
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Adicionar todos os testes
    suite.addTests(loader.loadTestsFromTestCase(TestPreprocessing))
    suite.addTests(loader.loadTestsFromTestCase(TestClassification))
    suite.addTests(loader.loadTestsFromTestCase(TestTemplateReply))
    suite.addTests(loader.loadTestsFromTestCase(TestGenerateReply))
    suite.addTests(loader.loadTestsFromTestCase(TestFlaskRoutes))
    suite.addTests(loader.loadTestsFromTestCase(TestIntegration))
    suite.addTests(loader.loadTestsFromTestCase(TestEdgeCases))
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    print("\n" + "="*70)
    print("📊 RESUMO DOS TESTES")
    print("="*70)
    print(f"✅ Testes executados: {result.testsRun}")
    print(f"✅ Sucessos: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"❌ Falhas: {len(result.failures)}")
    print(f"⚠️  Erros: {len(result.errors)}")
    print("="*70 + "\n")
    
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_tests_with_report()
    sys.exit(0 if success else 1)
