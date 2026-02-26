import cv2
import mediapipe as mp
import csv
import os
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# Caminho do modelo
model_path = "hand_landmarker.task"

# Configuração do MediaPipe
base_options = python.BaseOptions(model_asset_path=model_path)
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=1
)

detector = vision.HandLandmarker.create_from_options(options)

cap = cv2.VideoCapture(0)

# Criar arquivo se não existir
arquivo = "dataset.csv"
if not os.path.exists(arquivo):
    with open(arquivo, mode="w", newline="") as f:
        writer = csv.writer(f)
        header = []
        for i in range(21):
            header += [f"x{i}", f"y{i}", f"z{i}"]
        header.append("label")
        writer.writerow(header)

print("Pressione a letra desejada no teclado para rotular.")
print("Pressione ESC para sair.")

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

    if result.hand_landmarks:
        for hand_landmarks in result.hand_landmarks:

            # Normalização pelo punho
            base_x = hand_landmarks[0].x
            base_y = hand_landmarks[0].y

            dados = []

            for landmark in hand_landmarks:
                dados.append(landmark.x - base_x)
                dados.append(landmark.y - base_y)
                dados.append(landmark.z)

            tecla = cv2.waitKey(1) & 0xFF

            if tecla != 255 and tecla != 27:
                letra = chr(tecla).upper()

                with open(arquivo, mode="a", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow(dados + [letra])

                print(f"Salvo: {letra}")

            if tecla == 27:
                break

    cv2.imshow("Coleta de Dados", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()