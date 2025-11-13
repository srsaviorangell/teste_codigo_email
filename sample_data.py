"""
Dados de exemplo para treinamento do modelo de classificação de emails.
Contém exemplos reais de emails produtivos e improdutivos.
"""

TRAINING_DATA = [
    ("Olá, preciso de suporte técnico urgente. Sistema fora do ar desde as 14h. Ticket #12345", 1),
    ("Solicitamos atualização do status do projeto. Prazo de entrega é amanhã. Qual é a situação?", 1),
    ("Dúvida sobre como usar a API de integração. Poderia enviar a documentação?", 1),
    ("Temos um bug crítico em produção. Precisamos de correção imediata.", 1),
    ("Solicitação de acesso ao banco de dados para análise de dados.", 1),
    ("Cliente solicitou alteração no contrato. Precisamos revisar e aprovar.", 1),
    ("Qual é o status da implementação da feature X? Está no prazo?", 1),
    ("Preciso de informações sobre o novo sistema de faturamento.", 1),
    ("Reunião agendada para amanhã às 10h para discussão do roadmap.", 1),
    ("Relatório mensal pronto. Poderiam revisar antes de enviar ao cliente?", 1),
    ("Candidato interessado em vaga de desenvolvedor. Segue currículo em anexo.", 1),
    ("Necessário aprovação orçamentária para compra de licenças.", 1),
    ("Problema com integração do PayPal. Transações não estão sendo processadas.", 1),
    ("Solicitação de backup dos dados do mês de outubro.", 1),
    ("Feedback do cliente sobre a última entrega. Existem pontos para melhoria.", 1),
    
    ("Feliz aniversário! Tudo de bom para você neste dia especial!", 0),
    ("Obrigado pela oportunidade de trabalhar com vocês!", 0),
    ("Parabéns pelo ótimo desempenho no projeto!", 0),
    ("Muito obrigado pela ajuda. Ficou perfeito!", 0),
    ("Agradecimentos especiais ao time todo pela dedicação.", 0),
    ("Ótima apresentação ontem! Muito bem executada.", 0),
    ("Sucesso nos próximos passos do projeto!", 0),
    ("Valeu pela contribuição significativa ao projeto!", 0),
    ("Tudo bem? Só passando para dar um oi.", 0),
    ("Boa semana para você e sua família!", 0),
    ("Aproveita o fim de semana!", 0),
    ("Que tenha um ótimo dia!", 0),
    ("Sucesso em tudo que fizer!", 0),
    ("Tudo de melhor para você!", 0),
    ("Muito obrigado pela atenção!", 0),
]

TEXTS = [item[0] for item in TRAINING_DATA]
LABELS = [item[1] for item in TRAINING_DATA]