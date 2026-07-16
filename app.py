from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np
import os

# Inicializar Flask
app = Flask(__name__)

# Configurar CORS (permitir requisições do frontend)
CORS(app, resources={r"/*": {"origins": "*"}})

# Caminhos para os arquivos estáticos
MODEL_PATH = os.path.join("static", "modelo_alfabeto.pkl")
HAND_LANDMARKER_PATH = os.path.join("static", "hand_landmarker.task")

# Carregar o modelo treinado
try:
    modelo = joblib.load(MODEL_PATH)
    print("Modelo carregado com sucesso!")
except Exception as e:
    print(f"Erro ao carregar o modelo: {e}")
    modelo = None

# Rota de teste
@app.route("/")
def home():
    return jsonify({"status": "API funcionando!", "message": "Envie landmarks para /prever"})

# Rota principal para predição
@app.route("/prever", methods=["POST"])
def prever():
    try:
        if modelo is None:
            return jsonify({"erro": "Modelo não carregado"}), 500

        data = request.json
        if not data or "landmarks" not in data:
            return jsonify({"erro": "Dados inválidos. Envie um JSON com 'landmarks'"}), 400

        landmarks = np.array(data["landmarks"], dtype=np.float32)

        # Validar tamanho (21 landmarks × 3 coordenadas = 63 valores)
        if len(landmarks) != 63:
            return jsonify({
                "erro": f"Número inválido de landmarks. Esperado: 63 valores (21 landmarks). Recebido: {len(landmarks)}"
            }), 400

        # Normalização (igual ao treino)
        base_x, base_y, base_z = landmarks[0], landmarks[1], landmarks[2]
        dados_normalizados = []
        for i in range(0, len(landmarks), 3):
            x, y, z = landmarks[i], landmarks[i+1], landmarks[i+2]
            dados_normalizados.append(x - base_x)
            dados_normalizados.append(y - base_y)
            dados_normalizados.append(z - base_z)

        # Predição
        predicao = modelo.predict([dados_normalizados])[0]
        return jsonify({"predicao": predicao})

    except Exception as e:
        print(f"Erro na predição: {e}")
        return jsonify({"erro": f"Erro interno: {str(e)}"}), 500

# Iniciar a aplicação
if __name__ == "__main__":
    print("Iniciando a API Flask em http://localhost:5000")
    app.run(debug=True, host="0.0.0.0", port=5000)