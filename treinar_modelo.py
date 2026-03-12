import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report
import joblib
import seaborn as sns
import matplotlib.pyplot as plt

# --- CARREGAMENTO DO DATASET ---
dados = pd.read_csv("dataset.csv")

# --- LIMPEZA E SEPARAÇÃO ---
# X contém as features. O drop('label') remove a coluna de texto.
# O apply(pd.to_numeric) garante que tudo em X seja número.
X = dados.drop("label", axis=1).apply(pd.to_numeric, errors='coerce')
y = dados["label"]

# Remove linhas que possam ter gerado erro de conversão (NaN) ou infinitos
# (Isso resolve o erro de 'Value too large for dtype float32')
indices_validos = X.notna().all(axis=1) & ~np.isinf(X).any(axis=1)
X = X[indices_validos]
y = y[indices_validos]

# --- VALIDAÇÃO CRUZADA ---
# Avalia a robustez do modelo em 5 dobras antes do treino final
modelo = RandomForestClassifier(n_estimators=200, random_state=42)
scores = cross_val_score(modelo, X, y, cv=5)
print(f"Acurácia Média (Validação Cruzada): {scores.mean() * 100:.2f}%")
print(f"Desvio Padrão: {scores.std() * 100:.2f}%")

# --- DIVISÃO DOS DADOS ---
# O parâmetro stratify=y garante que o suporte seja equilibrado entre as classes
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# --- TREINAMENTO DO MODELO ---
modelo.fit(X_train, y_train)

# --- AVALIAÇÃO DE DESEMPENHO ---
acuracia = modelo.score(X_test, y_test)
print(f"\nAcurácia no Conjunto de Teste: {acuracia * 100:.2f}%")

# >>> INÍCIO DA MATRIZ DE CONFUSÃO <<<
y_pred = modelo.predict(X_test)
plt.figure(figsize=(10, 8))
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=sorted(y_test.unique()), 
            yticklabels=sorted(y_test.unique()))
plt.title('Matriz de Confusão - Classificação de LIBRAS')
plt.xlabel('Predito')
plt.ylabel('Real')
plt.savefig('matriz_confusao.png') # Salva a imagem
plt.show()

print("\nRelatório Técnico de Classificação:")
print(classification_report(y_test, y_pred))
# >>> FIM DA MATRIZ DE CONFUSÃO <<<

# --- EXPORTAÇÃO ---
joblib.dump(modelo, "modelo_alfabeto.pkl")
print("\nModelo salvo com sucesso: modelo_alfabeto.pkl")