#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Script de teste para visualizar os PRINTS DETALHADOS no console.
Mostra como cada email é analisado e qual é o score ajustado.
"""

import requests
import json

BASE_URL = "http://127.0.0.1:5000"

# Teste 1: Email Profissional Completo
print("\n" + "="*100)
print("[TESTE 1] EMAIL PROFISSIONAL COMPLETO")
print("="*100)

test1 = {
    "email_text": """
    Prezados,
    
    Estou enfrentando um ERRO CRÍTICO na integração com o sistema de backup de dados.
    A rotina automática não está funcionando desde ontem à noite e já perdemos informações.
    
    Preciso com URGÊNCIA de suporte técnico para resolver esse problema de acesso ao banco.
    Qual seria a solução mais rápida?
    
    Agradeço muito a atenção e fico no aguardo da resposta.
    
    Atenciosamente,
    Maria Silva
    """,
    "name": "Maria Silva",
    "email": "maria.silva@empresa.com.br",
    "subject": "URGENTE: Erro crítico na integração - Suporte"
}

response = requests.post(f"{BASE_URL}/process", json=test1)
print(f"\nStatus: {response.status_code}")
result = response.json()
print(f"Categoria: {result['category']}")
print(f"Score: {result['score']:.2f}")
print(f"Resposta: {result['reply']}\n")

# Teste 2: Email Spam/Suspeito
print("\n" + "="*100)
print("[TESTE 2] EMAIL SPAM/SUSPEITO")
print("="*100)

test2 = {
    "email_text": """
    CLIQUE AQUI!!! GANHE DINHEIRO RÁPIDO!!!
    
    Você foi selecionado para receber CASHBACK AUTOMÁTICO!!!
    Não perca essa oportunidade ÚNICA!!!
    
    ATIVE AGORA e receba BÔNUS IMEDIATO!!!
    """,
    "name": "xyz",
    "email": "noseiemail123",
    "subject": "CLIQUE AQUI!!! GANHE DINHEIRO!!!"
}

response = requests.post(f"{BASE_URL}/process", json=test2)
print(f"\nStatus: {response.status_code}")
result = response.json()
print(f"Categoria: {result['category']}")
print(f"Score: {result['score']:.2f}")
print(f"Resposta: {result['reply']}\n")

# Teste 3: Email Genérico/Curto
print("\n" + "="*100)
print("[TESTE 3] EMAIL GENÉRICO/CURTO")
print("="*100)

test3 = {
    "email_text": "olá tudo bem?",
    "name": "João Silva",
    "email": "joao@gmail.com",
    "subject": "oi"
}

response = requests.post(f"{BASE_URL}/process", json=test3)
print(f"\nStatus: {response.status_code}")
result = response.json()
print(f"Categoria: {result['category']}")
print(f"Score: {result['score']:.2f}")
print(f"Resposta: {result['reply']}\n")

# Teste 4: Email Dúvida Legítima
print("\n" + "="*100)
print("[TESTE 4] EMAIL COM DÚVIDA LEGÍTIMA (Muitas interrogações)")
print("="*100)

test4 = {
    "email_text": """
    Olá, tenho dúvidas sobre o sistema de relatório.
    Como faço para acessar os dados históricos? Qual é o caminho correto?
    Posso gerar relatórios em PDF? O sistema suporta exportação em Excel também?
    
    Preciso dessa informação para apresentar hoje à gerência.
    
    Obrigado pela atenção.
    """,
    "name": "Pedro Costa",
    "email": "pedro@empresa.com.br",
    "subject": "Dúvidas sobre sistema de relatório"
}

response = requests.post(f"{BASE_URL}/process", json=test4)
print(f"\nStatus: {response.status_code}")
result = response.json()
print(f"Categoria: {result['category']}")
print(f"Score: {result['score']:.2f}")
print(f"Resposta: {result['reply']}\n")

print("\n" + "="*100)
print("[TESTES COMPLETOS] Verifique o console do servidor para ver todos os PRINTS detalhados!")
print("="*100 + "\n")
