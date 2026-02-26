import cv2
import mediapipe as mp
import joblib
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

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

            base_x = hand_landmarks[0].x
            base_y = hand_landmarks[0].y

            dados = []

            for landmark in hand_landmarks:
                dados.append(landmark.x - base_x)
                dados.append(landmark.y - base_y)
                dados.append(landmark.z)

            predicao = modelo.predict([dados])
            letra = predicao[0]

            cv2.putText(frame, letra,
                        (50, 100),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        3, (0,255,0), 4)

    cv2.imshow("Reconhecimento Libras", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()