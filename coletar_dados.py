import cv2  # Biblioteca para processamento de imagem e captura de vídeo
import mediapipe as mp  # Framework para soluções de visão computacional (ML)
import csv  # Manipulação de arquivos de dados estruturados
import os  # Interface com o sistema operacional para gestão de arquivos
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import time

# --- CONFIGURAÇÃO DO MODELO---
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

# --- CONFIGURAÇÕES DE COLETA POR TEMPO ---
duracao_gravacao = 2.0  # Tempo de captura para cada clique (segundos)
frames_por_segundo_desejados = 10 # Quantos frames salvar por segundo (evita arquivos pesados)
intervalo_frames = 1 / frames_por_segundo_desejados
ultimo_frame_salvo = 0
gravando = False
letra_atual = ""
inicio_tempo = 0

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

print("Pressione a letra desejada para iniciar a gravação de 2 segundos.")
print("Movimente a mão levemente (perto/longe/ângulos) durante a gravação.")
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
        for idx, hand_landmarks in enumerate(result.hand_landmarks):

            # Descobrir se é esquerda ou direita
            label = result.handedness[idx][0].category_name
            
            # Inverte a lógica por causa do cv2.flip(frame, 1)
            if label == "Left":
                hand_label = "Right"
                cor = (0, 255, 0) # Verde para Direita
            else:
                hand_label = "Left"
                cor = (255, 0, 0) # Azul para Esquerda

            for landmark in hand_landmarks:
                x = int(landmark.x * frame.shape[1])
                y = int(landmark.y * frame.shape[0])
                cv2.circle(frame, (x, y), 5, cor, -1)

            cv2.putText(frame, hand_label,
                        (10, 40 + idx*40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1, cor, 2)
            
            # --- PREPARAÇÃO DOS DADOS (NORMALIZAÇÃO) ---
            base_x = hand_landmarks[0].x
            base_y = hand_landmarks[0].y
            dados_frame = []

            for landmark in hand_landmarks:
                dados_frame.append(landmark.x - base_x)
                dados_frame.append(landmark.y - base_y)
                dados_frame.append(landmark.z)

            # --- LÓGICA DE CAPTURA INTELIGENTE ---
            tecla = cv2.waitKey(1) & 0xFF

            # Se apertar uma tecla e não estiver gravando, inicia o processo
            if tecla != 255 and tecla != 27 and not gravando:
                letra_atual = chr(tecla).upper()
                gravando = True
                inicio_tempo = time.time()
                ultimo_frame_salvo = 0
                print(f"Gravando '{letra_atual}'... Mova a mão!")

            # Se estiver no período de gravação
            if gravando:
                tempo_passado = time.time() - inicio_tempo
                
                if tempo_passado < duracao_gravacao:
                    # Salva apenas se passou o tempo do intervalo (ex: 0.1s para 10fps)
                    if tempo_passado - ultimo_frame_salvo >= intervalo_frames:
                        with open(arquivo, mode="a", newline="") as f:
                            writer = csv.writer(f)
                            writer.writerow(dados_frame + [letra_atual])
                        ultimo_frame_salvo = tempo_passado
                    
                    # Feedback visual na tela
                    cv2.rectangle(frame, (0, 0), (300, 60), (0, 0, 255), -1)
                    cv2.putText(frame, f"GRAVANDO {letra_atual}: {duracao_gravacao - tempo_passado:.1f}s", 
                                (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                else:
                    gravando = False
                    print(f"Fim da coleta para {letra_atual}.")

            if tecla == 27: # ESC dentro do loop de landmarks
                break

    # Exibe a interface visual para o usuário
    cv2.imshow("Coleta de Dados", frame)

    # Monitora a tecla ESC para encerrar o programa fora do loop de landmarks
    if cv2.waitKey(1) & 0xFF == 27:
        break

# --- FINALIZAÇÃO ---
cap.release()  # Libera o hardware da câmera
cv2.destroyAllWindows()  # Fecha todas as janelas do OpenCV