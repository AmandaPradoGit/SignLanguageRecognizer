import time
import numpy as np
import pandas as pd
import joblib
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report

SEED = 42
PONTAS_DEDOS = [4, 8, 12, 16, 20]   # polegar, indicador, médio, anelar, mindinho
EXECUTAR_CV = True                   # validação cruzada de 10 folds (demora mais)
TESTAR_ANGULO = False                # opcional: inclinação média da mão por letra


# --- FUNÇÕES DE FEATURE ENGINEERING ---

def renormalizar_escala(df):
    """Força a distância pulso->landmark 9 a valer 1. Corrige datasets externos
    que usaram outra convenção de escala."""
    df = df.copy()
    escala = np.sqrt(df['x9']**2 + df['y9']**2 + df['z9']**2).replace(0, 1e-6)
    for col in [c for c in df.columns if c[0] in ('x', 'y', 'z')]:
        df[col] = df[col] / escala
    return df


def adicionar_features_distancia(df):
    df = df.copy()
    for i in range(len(PONTAS_DEDOS)):
        for j in range(i + 1, len(PONTAS_DEDOS)):
            a, b = PONTAS_DEDOS[i], PONTAS_DEDOS[j]
            df[f'dist_{a}_{b}'] = np.sqrt((df[f'x{a}'] - df[f'x{b}'])**2 +
                                          (df[f'y{a}'] - df[f'y{b}'])**2 +
                                          (df[f'z{a}'] - df[f'z{b}'])**2)
    return df


def adicionar_features_orientacao_e_cruzamento(df):
    df = df.copy()
    df['angulo_polegar_indicador'] = np.arctan2(df['y4'] - df['y5'], df['x4'] - df['x5'])
    df['f_vs_t_sign_x'] = df['x4'] - df['x5']
    df['f_vs_t_sign_z'] = df['z4'] - df['z5']
    df['cruzamento_indicador_medio'] = df['x8'] - df['x12']
    return df


def adicionar_features_letra_i(df):
    df = df.copy()
    df['mindinho_extensao_y'] = df['y0'] - df['y20']
    df['mindinho_dist_pulso'] = np.sqrt(df['x20']**2 + df['y20']**2 + df['z20']**2)
    df['mindinho_vs_indicador_y'] = df['y8'] - df['y20']
    return df


def preparar_features(df):
    df = adicionar_features_distancia(df)
    df = adicionar_features_orientacao_e_cruzamento(df)
    df = adicionar_features_letra_i(df)
    return df


# --- FUNÇÕES AUXILIARES DE RELATÓRIO ---

def tabela_metricas(y_true, y_pred, labels):
    """Precisão, revocação, F1 e suporte por letra, com média macro no fim."""
    rep = classification_report(y_true, y_pred, labels=labels,
                                output_dict=True, zero_division=0)
    tab = pd.DataFrame({k: rep[k] for k in labels}).T[
        ['precision', 'recall', 'f1-score', 'support']]
    tab.loc['Média macro'] = [rep['macro avg']['precision'], rep['macro avg']['recall'],
                              rep['macro avg']['f1-score'], rep['macro avg']['support']]
    tab.columns = ['Precisão', 'Revocação', 'F1-score', 'Suporte']
    tab['Suporte'] = tab['Suporte'].astype(int)
    return tab.round(2)


def plotar_matriz(cm, linhas, colunas, titulo, arquivo, cmap, ocultar_zeros=False):
    """Matriz de confusão com fontes grandes, para ler no artigo.
    Por padrão exibe os zeros; ocultar_zeros=True os esconde."""
    anot = np.where(cm == 0, "", cm.astype(str)) if ocultar_zeros else cm.astype(str)
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(cm, annot=anot, fmt="", annot_kws={"size": 13}, cmap=cmap,
                xticklabels=colunas, yticklabels=linhas, ax=ax,
                cbar_kws={'label': 'Número de amostras'})
    ax.set_title(titulo, fontsize=16, pad=12)
    ax.set_xlabel('Classe predita', fontsize=15)
    ax.set_ylabel('Classe real', fontsize=15)
    ax.tick_params(axis='both', labelsize=14)
    ax.tick_params(axis='y', labelrotation=0)       # letras do eixo y na horizontal
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=12)
    cbar.set_label('Número de amostras', fontsize=14)
    plt.tight_layout()
    plt.savefig(arquivo, dpi=300)
    plt.show()

def top_confusoes(cm, linhas, colunas, n=10):
    itens = [(real, pred, int(cm[i, j]))
             for i, real in enumerate(linhas)
             for j, pred in enumerate(colunas)
             if real != pred and cm[i, j] > 0]
    return sorted(itens, key=lambda t: -t[2])[:n]


def angulo_mao(df):
    """Inclinação do vetor pulso -> ponto 9, em graus."""
    return np.degrees(np.arctan2(df['y9'] - df['y0'], df['x9'] - df['x0']))


# --- 1. DATASET PRÓPRIO: CARREGAMENTO E LIMPEZA ---
dados = pd.read_csv("dataset_limpo.csv")
X = dados.drop("label", axis=1).apply(pd.to_numeric, errors='coerce')
y = dados["label"]

validos = X.notna().all(axis=1) & ~np.isinf(X).any(axis=1)
print(f"Amostras removidas na limpeza: {(~validos).sum()} de {len(X)}")
X, y = X[validos], y[validos]
X = preparar_features(X)
print(f"Atributos usados: {X.shape[1]}")

# --- 2. DIVISÃO, TREINAMENTO E TESTE INTERNO ---
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=SEED, stratify=y)

modelo = RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=SEED)
modelo.fit(X_train, y_train)

todas = list(modelo.classes_)
y_pred = modelo.predict(X_test)
acc_int = (y_pred == y_test).mean()
print(f"\nAcurácia no teste interno (20%): {acc_int * 100:.2f}%")

tab_int = tabela_metricas(y_test, y_pred, todas)
tab_int.to_csv("metricas_interno.csv")
cm_int = confusion_matrix(y_test, y_pred, labels=todas)
plotar_matriz(cm_int, todas, todas,
              'Matriz de confusão no conjunto de teste do domínio próprio',
              'matriz_confusao_interno.png', 'Blues')
print("Principais confusões (interno):", top_confusoes(cm_int, todas, todas))

# --- 2b. VALIDAÇÃO CRUZADA (10 folds) ---
if EXECUTAR_CV:
    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=SEED)
    scores = cross_val_score(
        RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=SEED),
        X, y, cv=cv)
    print(f"Validação cruzada (10 folds): {scores.mean() * 100:.2f}% "
          f"± {scores.std() * 100:.2f}")

# --- 3. BENCHMARK EXTERNO ---
dados_ext = pd.read_csv("datasetbenchmarking.csv")
X_ext = dados_ext.drop("label", axis=1).apply(pd.to_numeric, errors='coerce')
y_ext = dados_ext["label"]

validos_ext = X_ext.notna().all(axis=1) & ~np.isinf(X_ext).any(axis=1)
print(f"\nAmostras removidas no externo: {(~validos_ext).sum()} de {len(X_ext)}")
X_ext, y_ext = X_ext[validos_ext], y_ext[validos_ext]
X_ext = preparar_features(renormalizar_escala(X_ext))

# --- 4. ALINHAMENTO DE CLASSES E AVALIAÇÃO ---
classes_comuns = sorted(set(todas) & set(y_ext.unique()))
ausentes = sorted(set(todas) - set(classes_comuns))
print(f"Letras avaliadas ({len(classes_comuns)}): {classes_comuns}")
print(f"Letras ausentes no benchmark: {ausentes}")

mask = y_ext.isin(classes_comuns)
X_ext_f, y_ext_f = X_ext[mask], y_ext[mask]
y_pred_ext = modelo.predict(X_ext_f)
acc_ext = (y_pred_ext == y_ext_f).mean()
print(f"\nAcurácia no benchmark externo (letras comuns): {acc_ext * 100:.2f}%")

tab_ext = tabela_metricas(y_ext_f, y_pred_ext, classes_comuns)
tab_ext.to_csv("metricas_externo.csv")

# Matriz 26 x 26: linhas = classe real, colunas = classe predita (zeros visíveis).
# As linhas das letras ausentes no benchmark ficam zeradas; suas colunas mostram
# as amostras que o modelo previu como essas letras.
cm_ext = confusion_matrix(y_ext_f, y_pred_ext, labels=todas)
plotar_matriz(cm_ext, todas, todas,
              'Matriz de confusão no benchmark externo',
              'matriz_confusao_externo.png', 'Reds')
print("Principais confusões (externo):", top_confusoes(cm_ext, todas, todas, 15))

# Contagem de erros
erros_total = int((y_pred_ext != y_ext_f.values).sum())
erros_fora = int(np.isin(y_pred_ext, ausentes).sum())
print(f"Total de erros no externo: {erros_total} de {len(y_ext_f)}")
print(f"Previstas como letras ausentes no benchmark: {erros_fora} "
      f"({erros_fora / erros_total * 100:.1f}% dos erros)")

# --- 5. TABELA COMBINADA (para o artigo) ---
comb = pd.concat([tab_int[['Precisão', 'Revocação', 'F1-score']],
                  tab_ext[['Precisão', 'Revocação', 'F1-score']]],
                 axis=1, keys=['Domínio próprio', 'Benchmark externo'])
comb.to_csv("metricas_combinadas.csv")
print("\n", comb.fillna('-').to_string())

# --- 6. OPCIONAL: INCLINAÇÃO DA MÃO POR LETRA (hipótese da orientação) ---
if TESTAR_ANGULO:
    letras = ['P', 'I', 'A', 'U', 'R', 'D']
    comp = pd.DataFrame({
        'próprio': angulo_mao(X).groupby(y).mean().reindex(letras),
        'externo': angulo_mao(X_ext).groupby(y_ext).mean().reindex(letras)}).round(1)
    print("\nInclinação média pulso->ponto 9 (graus):\n", comp)

# --- 7. EXPORTAÇÃO E TEMPO DE INFERÊNCIA ---
joblib.dump(modelo, "modelo_alfabeto.pkl")

modelo.set_params(n_jobs=1)
amostra = X_test.iloc[[0]]
tempos = []
for _ in range(200):
    t0 = time.perf_counter()
    modelo.predict(amostra)
    tempos.append((time.perf_counter() - t0) * 1000)
print(f"\nTempo de inferência do classificador: mediana {np.median(tempos):.1f} ms por amostra")
print("(não inclui captura nem MediaPipe)")