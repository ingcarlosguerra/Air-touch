import pyrealsense2 as rs
import numpy as np
import cv2
import pyautogui
import mediapipe as mp

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.48, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils
pipeline = rs.pipeline()
config = rs.config()
pyautogui.FAILSAFE = False
pipeline_wrapper = rs.pipeline_wrapper(pipeline)
pipeline_profile = config.resolve(pipeline_wrapper)
device = pipeline_profile.get_device()
device_product_line = str(device.get_info(rs.camera_info.product_line))
found_rgb = False
arranque = False  
for s in device.sensors:
    if s.get_info(rs.camera_info.name) == 'RGB Camera':
        found_rgb = True
        break

if not found_rgb:
    print("The demo requires Depth camera with Color sensor")
    exit(0)

if device_product_line == 'L500':
    config.enable_stream(rs.stream.color, 960, 540, rs.format.bgr8, 30)
else:
    config.enable_stream(rs.stream.color, 1920, 1080, rs.format.bgr8, 30)

profile = pipeline.start(config)
align_to = rs.stream.color
align = rs.align(align_to)

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
        frames = pipeline.wait_for_frames()
        aligned_frames = align.process(frames)
        color_frame = aligned_frames.get_color_frame()
        if not color_frame:
            continue
        color_image = np.asanyarray(color_frame.get_data())
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

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(color_image_for_drawing, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                    index_tip_landmark = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                    x_pixel1, y_pixel1 = int(index_tip_landmark.x * color_image.shape[1]), int(index_tip_landmark.y * color_image.shape[0])
                    cv2.putText(color_image, f'Pixel XY: {x_pixel1}, {y_pixel1}', (50,100),
                                cv2.FONT_HERSHEY_PLAIN, 3, (30,144,255), 2)
                    pyautogui.moveTo((x_pixel1 +1920),y_pixel1)
                    cv2.circle(color_image_for_drawing, (x_pixel1, y_pixel1), 15, (0, 255, 0), -1)

            cv2.imshow('grid', color_image_for_drawing)
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
    pipeline.stop()
