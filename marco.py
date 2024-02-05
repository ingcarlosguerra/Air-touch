import cv2
import numpy as np

# Función de callback para eventos del mouse
def on_mouse(event, x, y, flags, param):
    global corners, dragging, current_corner

    if event == cv2.EVENT_LBUTTONDOWN:
        # Verificar si se hizo clic cerca de alguna esquina
        for i, corner in enumerate(corners):
            if np.sqrt((corner[0] - x)**2 + (corner[1] - y)**2) < 10:
                dragging = True
                current_corner = i

    elif event == cv2.EVENT_LBUTTONUP:
        dragging = False

    elif event == cv2.EVENT_MOUSEMOVE:
        # Actualizar la posición de la esquina mientras se arrastra
        if dragging:
            corners[current_corner] = (x, y)

# Inicializar variables
corners = [(100, 100), (300, 100), (300, 300), (100, 300)]  # Coordenadas iniciales de las esquinas
dragging = False  # Indica si se está arrastrando alguna esquina
current_corner = 0  # Índice de la esquina actual que se está moviendo

# Abrir la cámara
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    if not ret:
        print("Error al capturar el fotograma.")
        break

    # Dibujar el rectángulo con esquinas móviles
    cv2.polylines(frame, [np.array(corners)], isClosed=True, color=(255, 0, 0), thickness=2)

    # Mostrar las coordenadas de cada esquina en la ventana
    for i, corner in enumerate(corners):
        cv2.circle(frame, corner, 5, (0, 255, 0), -1)
        cv2.putText(frame, f'{corner}', (corner[0] + 10, corner[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1, cv2.LINE_AA)

    # Mostrar la imagen en la ventana
    cv2.imshow('Camera', frame)

    # Esperar 1 milisegundo y verificar si se presionó la tecla 'q' o 'Esc'
    key = cv2.waitKey(1)
    if key == ord('q') or key == 27:  # 27 es el código ASCII de la tecla 'Esc'
        break

# Liberar recursos
cap.release()
cv2.destroyAllWindows()

