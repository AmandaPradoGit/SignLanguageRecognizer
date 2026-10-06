import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report
import joblib
import seaborn as sns
import matplotlib.pyplot as plt

N_LANDMARKS = 21
LANDMARK_REFERENCIA = 9  # base do dedo médio, usado p/ escala
PONTAS_DEDOS = [4, 8, 12, 16, 20]  # polegar, indicador, médio, anelar, mindinho


# --- FUNÇÕES DE FEATURE ENGINEERING ---

def renormalizar_escala(df):
    """
    Força a distância pulso->landmark9 a valer 1 em todas as amostras.
    Corrige datasets externos que usaram outra convenção de escala.
    """
    df = df.copy()
    escala_atual = np.sqrt(df['x9']**2 + df['y9']**2 + df['z9']**2)
    escala_atual = escala_atual.replace(0, 1e-6)

    colunas_coord = [c for c in df.columns if c[0] in ('x', 'y', 'z')]
    for col in colunas_coord:
        df[col] = df[col] / escala_atual
    return df


def adicionar_features_distancia(df):
    """
    Adiciona distâncias euclidianas entre pares de pontas de dedo.
    Captura diretamente a abertura da mão e posição do polegar.
    """
    df = df.copy()
    for i in range(len(PONTAS_DEDOS)):
        for j in range(i + 1, len(PONTAS_DEDOS)):
            a, b = PONTAS_DEDOS[i], PONTAS_DEDOS[j]
            dx = df[f'x{a}'] - df[f'x{b}']
            dy = df[f'y{a}'] - df[f'y{b}']
            dz = df[f'z{a}'] - df[f'z{b}']
            df[f'dist_{a}_{b}'] = np.sqrt(dx**2 + dy**2 + dz**2)
    return df


def preparar_features(df):
    """Pipeline completo de feature engineering."""
    df = adicionar_features_distancia(df)
    return df


# --- 1. CARREGAMENTO E LIMPEZA (DATASET PRÓPRIO) ---
dados = pd.read_csv("dataset_limpo.csv")

X = dados.drop("label", axis=1).apply(pd.to_numeric, errors='coerce')
y = dados["label"]

indices_validos = X.notna().all(axis=1) & ~np.isinf(X).any(axis=1)
X = X[indices_validos]
y = y[indices_validos]

# Feature engineering no dataset de treino
X = preparar_features(X)

# --- 2. DIVISÃO E TREINAMENTO ---
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

modelo = RandomForestClassifier(
    n_estimators=300,
    class_weight='balanced',
    n_jobs=-1,
    random_state=42
)
modelo.fit(X_train, y_train)

acuracia_interna = modelo.score(X_test, y_test)
print(f"Acurácia no Teste Interno (Seus dados): {acuracia_interna * 100:.2f}%")


# =====================================================================
# >>> VALIDAÇÃO DE GENERALIZAÇÃO (KAGGLE) <<<
# =====================================================================

# --- 3. CARREGAMENTO DOS DADOS EXTERNOS ---
dados_kaggle = pd.read_csv("datasetbenchmarking.csv")

X_ext = dados_kaggle.drop("label", axis=1).apply(pd.to_numeric, errors='coerce')
y_ext = dados_kaggle["label"]

indices_validos_ext = X_ext.notna().all(axis=1) & ~np.isinf(X_ext).any(axis=1)
X_ext = X_ext[indices_validos_ext]
y_ext = y_ext[indices_validos_ext]

# Renormalização e Feature Engineering no Kaggle
X_ext = renormalizar_escala(X_ext)
X_ext = preparar_features(X_ext)

# --- 4. ALINHAMENTO DE CLASSES E AVALIAÇÃO ---

# Identifica apenas as letras efetivamente PRESENTES no dataset do Kaggle (> 0 amostras)
contagem_kaggle = y_ext.value_counts()
letras_validas_kaggle = contagem_kaggle[contagem_kaggle > 0].index.tolist()

# Interseção entre as classes treinadas e as letras presentes no Kaggle
classes_comuns = sorted(list(set(modelo.classes_) & set(letras_validas_kaggle)))

print(f"\n[ALINHAMENTO DE DATASETS]")
print(f"Letras avaliadas na interseção ({len(classes_comuns)} classes): {classes_comuns}")

# Filtra o Kaggle para conter estritamente as classes comuns
mascara_comuns = y_ext.isin(classes_comuns)
X_ext_filtrado = X_ext[mascara_comuns]
y_ext_filtrado = y_ext[mascara_comuns]

# Predição nos dados filtrados
y_pred_ext = modelo.predict(X_ext_filtrado)

# Métricas ajustadas
acuracia_ext = modelo.score(X_ext_filtrado, y_ext_filtrado)
print(f"\n[VALIDAÇÃO DE GENERALIZAÇÃO AJUSTADA]")
print(f"Acurácia no Dataset Externo (Apenas Letras Comuns): {acuracia_ext * 100:.2f}%")

print("\nRelatório de Classificação - Dados Externos (Kaggle):")
print(classification_report(y_ext_filtrado, y_pred_ext, labels=classes_comuns, zero_division=0))

# Matriz de Confusão com eixos perfeitamente alinhados (classes_comuns no X e no Y)
plt.figure(figsize=(10, 8))
cm_ext = confusion_matrix(y_ext_filtrado, y_pred_ext, labels=classes_comuns)
sns.heatmap(cm_ext, annot=True, fmt='d', cmap='Reds',
            xticklabels=classes_comuns,
            yticklabels=classes_comuns)
plt.title('Matriz de Confusão - Generalização Ajustada (Dataset Kaggle)')
plt.xlabel('Predito pelo Modelo')
plt.ylabel('Real (Kaggle)')
plt.tight_layout()
plt.savefig('matriz_confusao_kaggle.png')
plt.show()

# --- 5. EXPORTAÇÃO ---
joblib.dump(modelo, "modelo_alfabeto.pkl")
print("\nModelo treinado com seus dados e validado com Kaggle salvo com sucesso!")