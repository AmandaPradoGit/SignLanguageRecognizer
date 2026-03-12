import cv2
import mediapipe as mp
import joblib
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from collections import Counter # Para contar as predições mais frequentes

# Carregar modelo treinado
modelo = joblib.load("modelo_alfabeto.pkl")

# Configurar MediaPipe
model_path = "hand_landmarker.task"
base_options = python.BaseOptions(model_asset_path=model_path)
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=1
)

detector = vision.HandLandmarker.create_from_options(options)

# --- CONFIGURAÇÃO DE ESTABILIDADE ---
buffer_predicoes = []
tamanho_buffer = 10  # Quantos frames analisar para confirmar a letra
letra_estavel = ""

cap = cv2.VideoCapture(0)

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
            # Normalização (mesma do treino)
            base_x = hand_landmarks[0].x
            base_y = hand_landmarks[0].y
            dados = []

            for landmark in hand_landmarks:
                dados.append(landmark.x - base_x)
                dados.append(landmark.y - base_y)
                dados.append(landmark.z)

            # Predição instantânea
            predicao_imediata = modelo.predict([dados])[0]
            
            # Adiciona ao buffer para estabilizar
            buffer_predicoes.append(predicao_imediata)
            if len(buffer_predicoes) > tamanho_buffer:
                buffer_predicoes.pop(0)

            # A letra exibida será a que mais apareceu no buffer (Moda)
            letra_estavel = Counter(buffer_predicoes).most_common(1)[0][0]

            # Desenha um feedback visual (caixa ao redor da letra)
            cv2.rectangle(frame, (35, 20), (145, 130), (0, 255, 0), -1) # Fundo verde
            cv2.putText(frame, letra_estavel,
                        (55, 110),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        3, (255, 255, 255), 6) # Letra branca
            
            # Desenha os pontos da mão
            for landmark in hand_landmarks:
                x = int(landmark.x * frame.shape[1])
                y = int(landmark.y * frame.shape[0])
                cv2.circle(frame, (x, y), 3, (0, 255, 0), -1)

    else:
        buffer_predicoes.clear() # Limpa se a mão sair da tela

    # Interface estilizada
    cv2.putText(frame, "Reconhecimento em tempo real", (180, 40), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    cv2.imshow("Reconhecimento Libras", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()