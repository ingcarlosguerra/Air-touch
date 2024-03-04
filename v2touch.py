import cv2
import mediapipe as mp
import numpy as np
import pyrealsense2 as rs

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
pipeline.start(config)

try:
    while True:
        # Obtener frames de la cámara
        frames = pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        if not depth_frame or not color_frame:
            continue
        color_image = np.asanyarray(color_frame.get_data())
        color_image_for_drawing = color_image.copy()
        rgb_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_image)
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Dibujar los landmarks de la mano
                mp_drawing.draw_landmarks(color_image_for_drawing, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                index_tip_landmark = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                index_center_hand = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP]
                index_base_hand = hand_landmarks.landmark[mp_hands.WRIST]
                # x_pixel, y_pixel = int(index_tip_landmark.x * color_image.shape[1]), int(index_tip_landmark.y * color_image.shape[0])
                x_pixel, y_pixel = int(index_center_hand.x * color_image.shape[1]), int(index_center_hand.y * color_image.shape[0])
                x_pixel2, y_pixel2 = int(index_base_hand.x * color_image.shape[1]), int(index_base_hand.y * color_image.shape[0])
                try:
                    depth = depth_frame.get_distance(x_pixel, y_pixel)
                    depth2 = depth_frame.get_distance(x_pixel2, y_pixel2)
                except RuntimeError as e:
                    print(f"Error al obtener la distancia de profundidad: {e}")
                    depth = 0  # Asumir una profundidad de cero si hay un error

                # Mostrar las coordenadas en píxeles y la profundidad
                cv2.putText(color_image_for_drawing, f'Pixel XY: {x_pixel}, {y_pixel}', (x_pixel, y_pixel - 20),
                            cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 2)
                if depth > 0:  # Asegurarse de que la profundidad es válida
                    cv2.putText(color_image_for_drawing, f'Depth Z: {depth:.2f}m', (x_pixel, y_pixel - 40),
                                cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 2)
                    if depth > 3:  # Activar el "touch" si la distancia es mayor que 3 metros
                        # Dibujar un círculo verde en la posición del dedo índice
                        cv2.circle(color_image_for_drawing, (x_pixel, y_pixel), 15, (0, 255, 0), -1)

        # Mostrar la imagen resultante
        cv2.imshow('Hand Tracking', color_image_for_drawing)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    # Parar la transmisión y cerrar todas las ventanas
    pipeline.stop()
    hands.close()
    cv2.destroyAllWindows()



