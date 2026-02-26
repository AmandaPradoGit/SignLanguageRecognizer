import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib

# Carregar dataset
dados = pd.read_csv("dataset.csv")

X = dados.drop("label", axis=1)
y = dados["label"]

# Dividir treino/teste
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Criar modelo
modelo = RandomForestClassifier(n_estimators=200)
modelo.fit(X_train, y_train)

# Avaliação
acuracia = modelo.score(X_test, y_test)
print("Acurácia:", acuracia)

# Salvar modelo
joblib.dump(modelo, "modelo_alfabeto.pkl")
print("Modelo salvo como modelo_alfabeto.pkl")