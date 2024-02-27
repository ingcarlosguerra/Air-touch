import cv2
import numpy as np
import pyrealsense2 as rs

# Función para encontrar el contorno de la mano
def find_hand_contour(depth_frame):
    # Convierte el fotograma de profundidad a una matriz numpy
    depth_image = np.asanyarray(depth_frame.get_data())

    # Asegura que la imagen de profundidad sea de tipo uint8
    depth_image = np.uint8(depth_image)

    # Aplica un umbral para resaltar la mano
    _, thresholded = cv2.threshold(depth_image, 1000, 255, cv2.THRESH_BINARY)

    # Encuentra contornos en la imagen umbralizada
    contours, _ = cv2.findContours(thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Devuelve el contorno de la mano (seleccionamos el contorno más grande)
    hand_contour = max(contours, key=cv2.contourArea, default=None)

    return hand_contour

# Configura el objeto de configuración de la cámara RealSense
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

# Inicializa la cámara RealSense
pipeline = rs.pipeline()
pipeline.start(config)

try:
    while True:
        # Espera a que haya un nuevo conjunto de fotogramas
        frames = pipeline.wait_for_frames()

        # Obtiene el fotograma de profundidad
        depth_frame = frames.get_depth_frame()

        if not depth_frame:
            continue

        # Encuentra el contorno de la mano
        hand_contour = find_hand_contour(depth_frame)

        if hand_contour is not None:
            # Calcula el centroide del contorno de la mano
            M = cv2.moments(hand_contour)
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])

            # Obtiene la distancia al centroide de la mano
            distance = depth_frame.get_distance(cx, cy)

            # Dibuja el contorno de la mano y muestra la distancia en la consola
            cv2.drawContours(depth_image, [hand_contour], -1, (0, 255, 0), 2)
            cv2.circle(depth_image, (cx, cy), 5, (0, 0, 255), -1)
            print(f"Distancia a la mano: {distance:.3f} metros")

        # Muestra la imagen de profundidad
        cv2.imshow("Depth Image", depth_image)

        # Rompe el bucle si se presiona la tecla 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    # Cierra la conexión con la cámara RealSense
    pipeline.stop()
    cv2.destroyAllWindows()


