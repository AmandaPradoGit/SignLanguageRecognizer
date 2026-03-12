import pandas as pd

# Carrega o dataset
df = pd.read_csv("dataset.csv")

print(f"Total de linhas: {len(df)}")
print(f"Colunas encontradas: {df.columns.tolist()}\n")

# Verifica se existe alguma letra escondida nas colunas de coordenadas (X, Y, Z)
features = df.drop("label", axis=1)
for col in features.columns:
    # Tenta converter a coluna para número; se falhar, retorna o que não é número
    invalidos = df[pd.to_numeric(df[col], errors='coerce').isna()]
    
    if not invalidos.empty:
        print(f"⚠️ Erro na coluna [{col}]:")
        for idx, row in invalidos.iterrows():
            print(f"   -> Linha {idx}: Valor encontrado foi '{row[col]}'")

print("\nBusca finalizada.")