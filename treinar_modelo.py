import pandas as pd  # Manipulação e análise de dados estruturados
from sklearn.model_selection import train_test_split  # Divisão do dataset para validação
from sklearn.ensemble import RandomForestClassifier  # Algoritmo de aprendizado supervisionado
from sklearn.metrics import confusion_matrix, classification_report
import joblib  # Serialização de objetos Python (salvar o modelo)
import seaborn as sns  # para o gráfico ficar bonito
import matplotlib.pyplot as plt  # para exibir o gráfico

# --- CARREGAMENTO DO DATASET ---
# Lê o arquivo CSV contendo as coordenadas normalizadas e as etiquetas
dados = pd.read_csv("dataset.csv")

# Separação das variáveis: 
# X contém as "features" (coordenadas x, y, z de cada landmark)
# y contém o "target" (a letra ou classe correspondente)
X = dados.drop("label", axis=1)
y = dados["label"]

# --- DIVISÃO DOS DADOS ---
# Divide os dados em dois conjuntos: 80% para treinamento e 20% para teste/avaliação.
# random_state=42 garante a reprodutibilidade do experimento (divisão sempre igual).
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# --- TREINAMENTO DO MODELO ---
# Inicializa o classificador Random Forest (Floresta Aleatória) com 200 árvores de decisão.
# O uso de múltiplas árvores reduz o risco de overfitting e aumenta a robustez.
modelo = RandomForestClassifier(n_estimators=200)

# Ajusta o modelo aos dados de treino (processo de aprendizado)
modelo.fit(X_train, y_train)

# --- AVALIAÇÃO DE DESEMPENHO ---
# Calcula a acurácia média comparando as predições do modelo com os dados de teste (não vistos)
acuracia = modelo.score(X_test, y_test)
print(f"Acurácia Global do Modelo: {acuracia * 100:.2f}%")


# >>> INÍCIO DA MATRIZ DE CONFUSÃO <<<
# 1. O modelo faz previsões para os dados de teste
y_pred = modelo.predict(X_test)

# 2. Criação do gráfico da Matriz
plt.figure(figsize=(10, 8))
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=sorted(y_test.unique()), 
            yticklabels=sorted(y_test.unique()))

plt.title('Matriz de Confusão - Classificação de LIBRAS')
plt.xlabel('Predito (O que a IA disse)')
plt.ylabel('Real (O que você fez)')
plt.savefig('matriz_confusao.png') # Salva a imagem
plt.show()

# 3. Relatório detalhado (Precision, Recall e F1-Score)
print("\nRelatório Técnico de Classificação:")
print(classification_report(y_test, y_pred))
# >>> FIM DA MATRIZ DE CONFUSÃO <<<


# --- EXPORTAÇÃO (PERSISTÊNCIA) ---
# Salva o estado atual do modelo treinado em um arquivo binário (.pkl)
# Este arquivo será carregado no script de reconhecimento em tempo real (inferência).
joblib.dump(modelo, "modelo_alfabeto.pkl")
print("Modelo serializado com sucesso: modelo_alfabeto.pkl")