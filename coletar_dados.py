import cv2  # Biblioteca para processamento de imagem e captura de vídeo
import mediapipe as mp  # Framework para soluções de visão computacional (ML)
import csv  # Manipulação de arquivos de dados estruturados
import os  # Interface com o sistema operacional para gestão de arquivos
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# --- CONFIGURAÇÃO DO MODELO DE INFERÊNCIA ---
# Define o arquivo binário que contém a rede neural treinada para detecção de mãos
model_path = "hand_landmarker.task"

# Configura as opções do MediaPipe: aponta o modelo e define limites (apenas 1 mão)
base_options = python.BaseOptions(model_asset_path=model_path)
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=1
)

# Inicializa o detector de pontos de referência (landmarks) a partir das configurações
detector = vision.HandLandmarker.create_from_options(options)

# Inicializa a captura de vídeo através da webcam padrão (índice 0)
cap = cv2.VideoCapture(0)

# --- PREPARAÇÃO DO DATASET ---
arquivo = "dataset.csv"
# Se o arquivo não existir, cria e escreve o cabeçalho (Header)
if not os.path.exists(arquivo):
    with open(arquivo, mode="w", newline="") as f:
        writer = csv.writer(f)
        header = []
        # Gera rótulos para as coordenadas x, y, z de cada um dos 21 pontos da mão
        for i in range(21):
            header += [f"x{i}", f"y{i}", f"z{i}"]
        header.append("label") # Coluna de destino (target/classe)
        writer.writerow(header)

print("Pressione a letra desejada no teclado para rotular.")
print("Pressione ESC para sair.")

# --- LOOP PRINCIPAL DE PROCESSAMENTO ---
while True:
    ret, frame = cap.read() # Captura o frame da câmera
    if not ret:
        break

    # Pré-processamento: Inverte a imagem (espelhamento) e converte BGR para RGB (exigência do MediaPipe)
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Converte o frame em um objeto de imagem compatível com o MediaPipe Tasks
    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb_frame
    )

    # Executa a inferência para detectar a mão e seus pontos de referência
    result = detector.detect(mp_image)

    # Verifica se alguma mão foi detectada no frame atual
    if result.hand_landmarks:
        for hand_landmarks in result.hand_landmarks:

            # --- NORMALIZAÇÃO ESPACIAL (INVARIÂNCIA DE TRANSLAÇÃO) ---
            # Define o ponto 0 (pulso) como a origem (0,0) do sistema de coordenadas local
            base_x = hand_landmarks[0].x
            base_y = hand_landmarks[0].y

            dados = []

            # Extrai as coordenadas de cada ponto subtraindo a posição do pulso
            for landmark in hand_landmarks:
                dados.append(landmark.x - base_x)
                dados.append(landmark.y - base_y)
                dados.append(landmark.z)
            
            # --- CAPTURA DE TECLADO E ROTULAGEM ---
            tecla = cv2.waitKey(1) & 0xFF

            # Se uma tecla for pressionada (e não for ESC ou vazio)
            if tecla != 255 and tecla != 27:
                letra = chr(tecla).upper() # Converte o código da tecla para o caractere da letra

                # Gravação dos dados: Adiciona a linha de coordenadas + rótulo ao CSV
                with open(arquivo, mode="a", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow(dados + [letra])

                print(f"Salvo: {letra}")

            # Sai do loop interno se ESC for pressionado
            if tecla == 27:
                break

    # Exibe a interface visual para o usuário
    cv2.imshow("Coleta de Dados", frame)

    # Monitora a tecla ESC para encerrar o programa
    if cv2.waitKey(1) & 0xFF == 27:
        break

# --- FINALIZAÇÃO ---
cap.release()  # Libera o hardware da câmera
cv2.destroyAllWindows()  # Fecha todas as janelas do OpenCV