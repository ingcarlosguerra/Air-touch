import cv2
import time
cap = cv2.VideoCapture(1)
if not cap.isOpened():
    print("No se puede abrir la cámara")
    exit()

desired_width = 1920
desired_height = 1080
cap.set(cv2.CAP_PROP_FRAME_WIDTH, desired_width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, desired_height)
desired_fps = 30
cap.set(cv2.CAP_PROP_FPS, desired_fps)

while True:
    ret, frame = cap.read()
    if not ret:
        print("No se puede recibir el frame (la transmisión finalizó...). Saliendo...")
        break

    

    cv2.imshow('Webcam (Configurada)', frame)

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
