import numpy as np
import cv2
import pyautogui
import mediapipe as mp
import time
import cv2

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("No se puede abrir la cámara")
    exit()

desired_width = 1920
desired_height = 1080
cap.set(cv2.CAP_PROP_FRAME_WIDTH, desired_width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, desired_height)
desired_fps = 60
cap.set(cv2.CAP_PROP_FPS, desired_fps)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.40, min_tracking_confidence=0.40)
mp_drawing = mp.solutions.drawing_utils
pyautogui.FAILSAFE = False
arranque = False 

def find_centroid(contour):
    M = cv2.moments(contour)
    if M['m00'] == 0:
        return None
    cx = int(M['m10'] / M['m00'])
    cy = int(M['m01'] / M['m00'])
    return (cx, cy)


def on_mouse(event, x, y, flags, param):
    global corners, dragging, current_corner, arranque

    if event == cv2.EVENT_LBUTTONDOWN:
        for i, corner in enumerate(corners):
            if np.sqrt((corner[0] - x)**2 + (corner[1] - y)**2) < 10:
                dragging = True
                current_corner = i

        if button_x < x < button_x + button_w and button_y < y < button_y + button_h:
            print("El sistema Touch A iniciado")
            arranque = True

    elif event == cv2.EVENT_LBUTTONUP:
        dragging = False

    elif event == cv2.EVENT_MOUSEMOVE:
        if dragging:
            corners[current_corner] = (x, y)

def on_finish_button_click(event, x, y, flags, param):
    global is_finished
    if event == cv2.EVENT_LBUTTONDOWN:
        if finish_button_x < x < finish_button_x + finish_button_w and finish_button_y < y < finish_button_y + finish_button_h:
            print("Finish Button Clicked!")
            is_finished = True

corners = [(300, 100), (1600, 100), (1600, 900), (300, 900)]
dragging = False
current_corner = 0
cv2.namedWindow('Camera')
cv2.setMouseCallback('Camera', on_mouse)
button_x, button_y, button_w, button_h = 800, 960, 300, 50
finish_button_x, finish_button_y, finish_button_w, finish_button_h = 900, 600, 300, 50
is_finished = False

try:
    while not is_finished:
        ret,frame = cap.read()
        color_image = frame
        # color_image=  cv2.rotate(color_image, cv2.ROTATE_180)     ### camara rotada de RGB
        cv2.polylines(color_image, [np.array(corners)], isClosed=True, color=(0, 0, 255), thickness=2)
        corner1 = (corners[0][0], corners[0][1])
        corner2 = (corners[1][0], corners[1][1])
        corner3 = (corners[2][0], corners[2][1])
        corner4 = (corners[3][0], corners[3][1])
        npcorners = np.array([corner1, corner2, corner3, corner4], dtype="float32")
        widthA = np.sqrt(((corner3[0] - corner4[0]) ** 2) + ((corner3[1] - corner4[1]) ** 2))
        widthB = np.sqrt(((corner2[0] - corner1[0]) ** 2) + ((corner2[1] - corner1[1]) ** 2))
        heightA = np.sqrt(((corner2[0] - corner3[0]) ** 2) + ((corner2[1] - corner3[1]) ** 2))
        heightB = np.sqrt(((corner1[0] - corner4[0]) ** 2) + ((corner1[1] - corner4[1]) ** 2))
        maxWidth = max(int(widthA), int(widthB))
        maxHeight = max(int(heightA), int(heightB))
        dst = np.array([
            [0, 0],
            [maxWidth - 1, 0],
            [maxWidth - 1, maxHeight - 1],
            [0, maxHeight - 1]], dtype="float32")

        if arranque:
            imagen= color_image.copy()
            M = cv2.getPerspectiveTransform(npcorners, dst)
            grid = cv2.warpPerspective(imagen, M, (maxWidth, maxHeight))
            rgb_image = cv2.cvtColor(grid, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb_image)
            color_image_for_drawing = grid.copy()


            # Convertir la imagen a espacio de color HSV
            hsv = cv2.cvtColor(color_image_for_drawing, cv2.COLOR_BGR2HSV)

            # Definir rangos de color rojo en HSV
            # Nota: El rojo puede tener dos rangos debido a su posición en el espectro de color
            lower_red1 = np.array([0, 120, 70])
            upper_red1 = np.array([10, 255, 255])
            lower_red2 = np.array([170, 120, 70])
            upper_red2 = np.array([180, 255, 255])

            # Crear máscaras para color rojo
            mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
            mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
            mask = cv2.bitwise_or(mask1, mask2)

            # Aplicar la máscara a la imagen original
            red_regions = cv2.bitwise_and(color_image_for_drawing, color_image_for_drawing, mask=mask)

            # Encontrar contornos en la máscara
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # Suponiendo que queremos el contorno más grande si hay varios
            if contours:
                max_contour = max(contours, key=cv2.contourArea)
                centroid = find_centroid(max_contour)
                if centroid:
                    print("Centroide del contorno rojo:", centroid)
                    # Dibujar el contorno y el centroide en la imagen de contornos
                    cv2.drawContours(color_image_for_drawing, [max_contour], -1, (0, 255, 0), 2)  # Dibuja en verde para visibilidad
                    cv2.circle(color_image_for_drawing, centroid, 5, (255, 255, 255), -1)
                else:
                    print("No se pudo encontrar el centroide.")
            else:
                print("No se encontraron contornos rojos.")


            cv2.imshow('grid', mask)
            img_resized = cv2.resize(grid, (1920, 1080), interpolation=cv2.INTER_AREA)

        cv2.polylines(color_image, [np.array(corners)], isClosed=True, color=(255,144,30), thickness=2)
        cv2.putText(color_image, "Iniciar Touch", (button_x, button_y + 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5,(255,144,30), 2, cv2.LINE_AA)
        cv2.rectangle(color_image, (button_x, button_y), (button_x + button_w, button_y + button_h), (255,144,30), 2)
        cv2.imshow('Camera', color_image)

        key = cv2.waitKey(1)

        if key & 0xFF == ord('q') or key == 27:
            print(corners)
            cv2.destroyAllWindows()
            break

finally:
    cap.release()
    cv2.destroyAllWindows()
