from collections import Counter

texto = "chatbot chatbot inteligência artificial chatbot aprendizado"

# Transformar texto em lista de palavras
palavras = texto.split()

# Calcular frequência
frequencia = Counter(palavras)

print(frequencia)