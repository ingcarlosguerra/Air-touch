import cv2
import time

# Inicia la captura de video de la cámara
# Asume que la cámara es la primera cámara
cap = cv2.VideoCapture(1)

# Verifica si la cámara se inició correctamente
if not cap.isOpened():
    print("No se puede abrir la cámara")
    exit()

# desired_width = 1920
# desired_height = 1080
# cap.set(cv2.CAP_PROP_FRAME_WIDTH, desired_width)
# cap.set(cv2.CAP_PROP_FRAME_HEIGHT, desired_height)
# desired_fps = 60
# cap.set(cv2.CAP_PROP_FPS, desired_fps)

while True:
    # Captura frame-por-frame
    ret, frame = cap.read()

    # Si el frame se lee correctamente ret es True
    if not ret:
        print("No se puede recibir el frame (la transmisión finalizó...). Saliendo...")
        break

    # Muestra el frame resultante
    cv2.imshow('Webcam (Configurada)', frame)

    # Espera la tecla 'q' para salir del loop
    if cv2.waitKey(1) == ord('q'):
        break

# Cuando todo está hecho, libera los recursos
cap.release()
cv2.destroyAllWindows()
