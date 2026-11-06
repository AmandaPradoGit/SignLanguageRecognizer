import cv2  # Biblioteca para processamento de imagem e captura de vídeo
import mediapipe as mp  # Framework para soluções de visão computacional (ML)
import csv  # Manipulação de arquivos de dados estruturados
import os  # Interface com o sistema operacional para gestão de arquivos
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import time

# --- CONFIGURAÇÃO DO MODELO---
static_dir = os.path.join(os.path.dirname(__file__), "static")
model_path = os.path.join(static_dir, "hand_landmarker.task")

base_options = python.BaseOptions(model_asset_path=model_path)
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=1
)

detector = vision.HandLandmarker.create_from_options(options)

cap = cv2.VideoCapture(0)

# --- CONFIGURAÇÃO DE COLETA FIXA ---
TOTAL_FRAMES = 100
frames_coletados = 0

frames_por_segundo_desejados = 5
intervalo_frames = 1 / frames_por_segundo_desejados
ultimo_frame_salvo = 0

gravando = False
letra_atual = ""

LANDMARK_REFERENCIA = 9  # base do dedo médio, usado p/ normalizar escala

# --- PREPARAÇÃO DO DATASET ---
arquivo = "dataset_limpo.csv"

if not os.path.exists(arquivo):
    with open(arquivo, mode="w", newline="") as f:
        writer = csv.writer(f)
        header = []
        for i in range(21):
            header += [f"x{i}", f"y{i}", f"z{i}"]
        header.append("label")
        writer.writerow(header)

file = open(arquivo, mode="a", newline="")
writer = csv.writer(file)

print("Pressione uma tecla (A-Z)")
print("Pressione ESC para sair.")

# --- LOOP PRINCIPAL ---
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb_frame
    )

    result = detector.detect(mp_image)

    tecla = cv2.waitKey(1) & 0xFF

    if result.hand_landmarks:
        for idx, hand_landmarks in enumerate(result.hand_landmarks):

            # Descobrir se é esquerda ou direita (já corrigido pelo flip)
            label = result.handedness[idx][0].category_name
            if label == "Left":
                hand_label = "Right"
                cor = (0, 255, 0)
            else:
                hand_label = "Left"
                cor = (255, 0, 0)

            for landmark in hand_landmarks:
                x = int(landmark.x * frame.shape[1])
                y = int(landmark.y * frame.shape[0])
                cv2.circle(frame, (x, y), 5, cor, -1)

            cv2.putText(frame, hand_label,
                        (10, 40 + idx*40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1, cor, 2)

            # --- NORMALIZAÇÃO ---
            base_x = hand_landmarks[0].x
            base_y = hand_landmarks[0].y
            base_z = hand_landmarks[0].z

            # ponto de referência já com pulso zerado (pra calcular escala)
            ref = hand_landmarks[LANDMARK_REFERENCIA]
            escala = ((ref.x - base_x) ** 2 +
                      (ref.y - base_y) ** 2 +
                      (ref.z - base_z) ** 2) ** 0.5
            if escala < 1e-6:
                escala = 1e-6

            dados_frame = []

            for landmark in hand_landmarks:
                x_norm = (landmark.x - base_x) / escala
                y_norm = (landmark.y - base_y) / escala
                z_norm = (landmark.z - base_z) / escala

                # Padroniza handedness: tudo tratado como mão DIREITA.
                # Usa hand_label (já corrigido pelo flip) pra decidir.
                if hand_label == "Left":
                    x_norm = -x_norm

                dados_frame.append(x_norm)
                dados_frame.append(y_norm)
                dados_frame.append(z_norm)

            # --- INICIAR GRAVAÇÃO ---
            if tecla != 255 and tecla != 27 and not gravando:
                letra_atual = chr(tecla).upper()
                gravando = True
                frames_coletados = 0
                ultimo_frame_salvo = 0
                print(f"Gravando '{letra_atual}'...")

            # --- GRAVAÇÃO CONTROLADA ---
            if gravando:
                tempo_atual = time.time()

                if (tempo_atual - ultimo_frame_salvo) >= intervalo_frames:
                    writer.writerow(dados_frame + [letra_atual])
                    frames_coletados += 1
                    ultimo_frame_salvo = tempo_atual

                cv2.rectangle(frame, (0, 0), (320, 60), (0, 0, 255), -1)
                cv2.putText(frame,
                            f"{letra_atual}: {frames_coletados}/{TOTAL_FRAMES}",
                            (10, 40),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.8,
                            (255, 255, 255),
                            2)

                if frames_coletados >= TOTAL_FRAMES:
                    gravando = False
                    print(f"Coleta finalizada para '{letra_atual}'")

    cv2.imshow("Coleta de Dados", frame)

    if tecla == 27:
        break

# --- FINALIZAÇÃO ---
file.close()
cap.release()
cv2.destroyAllWindows()