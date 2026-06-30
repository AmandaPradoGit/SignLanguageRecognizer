from flask import Flask, request, jsonify
from flask_cors import CORS  # Para permitir requisições do frontend
import joblib  # Para carregar o modelo .pkl
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import os

# Inicializar a aplicação Flask
app = Flask(__name__)

# Configurar CORS para permitir requisições do frontend
CORS(app, resources={r"/*": {"origins": "*"}})

# Caminhos para os arquivos estáticos
MODEL_PATH = os.path.join("static", "modelo_alfabeto.pkl")
HAND_LANDMARKER_PATH = os.path.join("static", "hand_landmarker.task")

# Carregar o modelo treinado (Random Forest)
try:
    modelo = joblib.load(MODEL_PATH)
    print("Modelo carregado com sucesso!")
except Exception as e:
    print(f"Erro ao carregar o modelo: {e}")
    modelo = None

# Carregar o detector de landmarks das mãos (MediaPipe)
try:
    base_options = python.BaseOptions(model_asset_path=HAND_LANDMARKER_PATH)
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        num_hands=1  # Considerar apenas uma mão por frame
    )
    detector = vision.HandLandmarker.create_from_options(options)
    print(" Detector MediaPipe carregado com sucesso!")
except Exception as e:
    print(f" Erro ao carregar o detector MediaPipe: {e}")
    detector = None

# Rota para testar se a API está funcionando
@app.route("/")
def home():
    return jsonify({"status": "API funcionando!", "message": "Envie landmarks para /prever"})

# Rota principal para predição
@app.route("/prever", methods=["POST"])
def prever():
    try:
        # Verificar se o modelo e o detector foram carregados
        if modelo is None:
            return jsonify({"erro": "Modelo não carregado"}), 500
        if detector is None:
            return jsonify({"erro": "Detector MediaPipe não carregado"}), 500

        # Receber os landmarks do frontend
        data = request.json
        if not data or "landmarks" not in data:
            return jsonify({"erro": "Dados inválidos. Envie um JSON com 'landmarks'"}), 400

        landmarks = np.array(data["landmarks"], dtype=np.float32)

        # Verificar se o tamanho dos landmarks está correto (21 landmarks * 3 coordenadas = 63 valores)
        if len(landmarks) != 63:
            return jsonify({"erro": f"Tamanho inválido de landmarks. Esperado: 63, Recebido: {len(landmarks)}"}), 400

        # Normalização dos landmarks (igual ao treino)
        # Usar o primeiro landmark como base para normalização
        base_x, base_y, base_z = landmarks[0], landmarks[1], landmarks[2]
        dados_normalizados = []
        for i in range(0, len(landmarks), 3):
            x, y, z = landmarks[i], landmarks[i+1], landmarks[i+2]
            dados_normalizados.append(x - base_x)
            dados_normalizados.append(y - base_y)
            dados_normalizados.append(z - base_z)

        # Converter para array do numpy e fazer predição
        dados_normalizados = np.array(dados_normalizados).reshape(1, -1)
        predicao = modelo.predict(dados_normalizados)[0]

        # Retornar a predição
        return jsonify({"predicao": predicao})

    except Exception as e:
        print(f" Erro na predição: {e}")
        return jsonify({"erro": f"Erro interno: {str(e)}"}), 500

# Iniciar a aplicação
if __name__ == "__main__":
    print(" Iniciando a API...")
    app.run(debug=True, host="0.0.0.0", port=5000)