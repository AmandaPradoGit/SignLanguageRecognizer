import cv2 #importa a biblioteca openCV que acessa a câmera
import mediapipe as mp #biblioteca mediapipe para detectar a mão
from mediapipe.tasks import python #Esse import traz a parte base para configurar modelos.
from mediapipe.tasks.python import vision #importa o módulo específico de Visão Computacional.

# usa o modelo hand_landmark
model_path = "hand_landmarker.task"

# Criar opções do modelo
base_options = python.BaseOptions(model_asset_path=model_path)
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=2 #numero de mãos a serem detectadas
)

# Criar detector
detector = vision.HandLandmarker.create_from_options(options)

#utiliza a câmera do computador
cap = cv2.VideoCapture(0)
#captura os frames até ser encerrado
while True:
    ret, frame = cap.read()
    if not ret: # se a camera falhar o programa sai do loop
        break

    frame = cv2.flip(frame, 1) #espelhamento da imagem
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)#converte o BGR para RGB, padrão do mediapipe

    #colocando a imagem no padrão do mediapipe 
    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb_frame
    )

    #result é o retorno da IA depois de analisar a imagem, 
    #o result é um objeto que contem os landmarks
    result = detector.detect(mp_image)
        

    #verifica a mão detectada
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

    cv2.imshow("Teste MediaPipe", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()