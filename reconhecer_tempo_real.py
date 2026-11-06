import cv2
import mediapipe as mp
import joblib
import os
import time
import numpy as np
import psutil
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from collections import Counter, deque

# Carregar modelo treinado (do diretório static/)
static_dir = os.path.join(os.path.dirname(__file__), "static")
modelo_path = os.path.join(static_dir, "modelo_alfabeto.pkl")
modelo = joblib.load(modelo_path)

# Configurar MediaPipe (task file em static/)
model_path = os.path.join(static_dir, "hand_landmarker.task")
base_options = python.BaseOptions(model_asset_path=model_path)
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=1
)
detector = vision.HandLandmarker.create_from_options(options)


# --- EXTRAÇÃO DE ATRIBUTOS ---
# ATENÇÃO: esta função deve ser a MESMA usada no treino (pipeline único).
# Aqui está apenas a translação pelo pulso (63 atributos). Se o modelo final
# usa normalização por escala + 19 atributos geométricos (82 no total),
# substitua o corpo desta função pelo código desse pipeline.
def extrair_atributos(hand_landmarks):
    base_x = hand_landmarks[0].x
    base_y = hand_landmarks[0].y
    base_z = hand_landmarks[0].z
    dados = []
    for landmark in hand_landmarks:
        dados.append(landmark.x - base_x)
        dados.append(landmark.y - base_y)
        dados.append(landmark.z - base_z)
    return dados


# --- CONFIGURAÇÃO DE ESTABILIDADE ---
buffer_predicoes = []
tamanho_buffer = 10
letra_estavel = ""

buffer_posicoes = deque(maxlen=20)
buffer_tempos = deque(maxlen=20)

# --- MÉTRICAS DE DESEMPENHO ---
N_AQUECIMENTO = 30                  # quadros descartados no início
t_mp, t_feat, t_clf, t_pipeline = [], [], [], []   # só quadros com mão detectada (s)
t_loop = []                         # tempo do loop completo, todos os quadros (s)
proc = psutil.Process()
proc.cpu_percent(interval=None)     # "zera" o contador de CPU
ram_pico_mb = 0.0

cap = cv2.VideoCapture(0)

while True:
    t_ini_loop = time.perf_counter()

    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

    # Etapa 1: MediaPipe
    t0 = time.perf_counter()
    result = detector.detect(mp_image)
    t1 = time.perf_counter()

    if result.hand_landmarks:
        for hand_landmarks in result.hand_landmarks:
            # Etapa 2: atributos
            t2 = time.perf_counter()
            dados = extrair_atributos(hand_landmarks)
            t3 = time.perf_counter()

            # Etapa 3: classificação
            predicao_imediata = modelo.predict([dados])[0]
            t4 = time.perf_counter()

            t_mp.append(t1 - t0)
            t_feat.append(t3 - t2)
            t_clf.append(t4 - t3)
            t_pipeline.append((t1 - t0) + (t4 - t2))

            # --- Posição do indicador (landmark 8) para I/J e D/Z ---
            base_x = hand_landmarks[0].x
            base_y = hand_landmarks[0].y
            x8 = hand_landmarks[8].x - base_x
            y8 = hand_landmarks[8].y - base_y
            buffer_posicoes.append((x8, y8))
            buffer_tempos.append(time.time())

            buffer_predicoes.append(predicao_imediata)
            if len(buffer_predicoes) > tamanho_buffer:
                buffer_predicoes.pop(0)

            letra_base = Counter(buffer_predicoes).most_common(1)[0][0]
            letra_estavel = letra_base

            if len(buffer_posicoes) >= 10:
                pos_inicial = buffer_posicoes[0]
                pos_final = buffer_posicoes[-1]
                dx = pos_final[0] - pos_inicial[0]
                dy = pos_final[1] - pos_inicial[1]

                variacao_x = max(p[0] for p in buffer_posicoes) - min(p[0] for p in buffer_posicoes)
                variacao_y = max(p[1] for p in buffer_posicoes) - min(p[1] for p in buffer_posicoes)

                if letra_base.lower() in ['i', 'j']:
                    if dy > 0.03 and abs(dx) < 0.02:
                        letra_estavel = 'J'
                    elif variacao_y > 0.05:
                        letra_estavel = 'J'
                    else:
                        letra_estavel = 'I'
                elif letra_base.lower() in ['d', 'z']:
                    if abs(dx) > 0.04 and variacao_x > 0.05:
                        letra_estavel = 'Z'
                    else:
                        letra_estavel = 'D'

            # Feedback visual
            cv2.rectangle(frame, (35, 20), (145, 130), (0, 255, 0), -1)
            cv2.putText(frame, letra_estavel, (55, 110),
                        cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 255, 255), 6)

            for landmark in hand_landmarks:
                x = int(landmark.x * frame.shape[1])
                y = int(landmark.y * frame.shape[0])
                cv2.circle(frame, (x, y), 3, (0, 255, 0), -1)
    else:
        buffer_predicoes.clear()
        buffer_posicoes.clear()

    cv2.imshow("Reconhecimento Libras", frame)
    tecla = cv2.waitKey(1) & 0xFF

    # Tempo do loop completo (captura + pipeline + desenho + imshow)
    t_loop.append(time.perf_counter() - t_ini_loop)
    ram_pico_mb = max(ram_pico_mb, proc.memory_info().rss / 1e6)

    if tecla == 27:  # ESC
        break

cap.release()
cv2.destroyAllWindows()


# --- RELATÓRIO ---
def resumo(nome, v):
    v = np.array(v[N_AQUECIMENTO:]) * 1000  # ms
    if len(v) == 0:
        print(f"{nome}: sem dados suficientes")
        return
    print(f"{nome:<22} média {v.mean():6.2f} ms | desvio {v.std():5.2f} | "
          f"mediana {np.median(v):6.2f} | p95 {np.percentile(v, 95):6.2f}")


print("\n===== DESEMPENHO (quadros com mão detectada) =====")
resumo("MediaPipe", t_mp)
resumo("Atributos", t_feat)
resumo("Random Forest (predict)", t_clf)
resumo("Pipeline (sem desenho)", t_pipeline)

print("\n===== LOOP COMPLETO (todos os quadros) =====")
resumo("Loop completo", t_loop)
loop = np.array(t_loop[N_AQUECIMENTO:])
if len(loop) > 0:
    print(f"FPS médio: {1.0 / loop.mean():.1f}")

print("\n===== RECURSOS =====")
print(f"CPU média do processo: {proc.cpu_percent(interval=None):.1f}% "
      f"(pode passar de 100% em múltiplos núcleos)")
print(f"RAM pico: {ram_pico_mb:.0f} MB")
print(f"Tamanho do modelo: {os.path.getsize(modelo_path) / 1e6:.1f} MB")
print(f"Quadros com mão analisados: {max(len(t_pipeline) - N_AQUECIMENTO, 0)}")