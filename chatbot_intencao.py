from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

frases = [
    "quero comprar produto",
    "desejo fazer compra",
    "quero cancelar pedido",
    "cancelar minha compra",
    "qual horário funcionamento",
    "vocês abrem sábado",
    "quero saber horário",
    "qual o horário da loja"
]

intencoes = [
    "comprar",
    "comprar",
    "cancelar",
    "cancelar",
    "horario",
    "horario",
    "horario",
    "horario"
]

# criar vetorizador
vetorizador = CountVectorizer()

# transformar frases em vetores
X = vetorizador.fit_transform(frases)

# treinar modelo
modelo = MultinomialNB()
modelo.fit(X, intencoes)

# pedir frase do usuário
entrada = input("Digite sua mensagem: ")

# transformar entrada em vetor
entrada_vetor = vetorizador.transform([entrada])

# prever intenção
predicao = modelo.predict(entrada_vetor)

print("Intenção detectada:", predicao[0])