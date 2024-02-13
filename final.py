import pyrealsense2 as rs
import numpy as np
import cv2
import pyautogui

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

config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

if device_product_line == 'L500':
    config.enable_stream(rs.stream.color, 960, 540, rs.format.bgr8, 30)
else:
    config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)

profile = pipeline.start(config)

depth_sensor = profile.get_device().first_depth_sensor()
depth_scale = depth_sensor.get_depth_scale()

clipping_distance_in_meters_max = 1.2  # metros  # Distancia maxima a la pared sin guenerar ruido
clipping_distance_in_meters_min = 1.1  # metros   # se recomienda dar unos 10 cm de la pared

clipping_distance_max = clipping_distance_in_meters_max / depth_scale
clipping_distance_min = clipping_distance_in_meters_min / depth_scale

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

corners = [(100, 100), (300, 100), (300, 300), (100, 300)]
dragging = False
current_corner = 0

cv2.namedWindow('Camera')
cv2.setMouseCallback('Camera', on_mouse)
button_x, button_y, button_w, button_h = 500, 600, 300, 50
finish_button_x, finish_button_y, finish_button_w, finish_button_h = 900, 600, 300, 50

is_finished = False

try:
    while not is_finished:
        frames = pipeline.wait_for_frames()
        aligned_frames = align.process(frames)
        aligned_depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()

        if not aligned_depth_frame or not color_frame:
            continue

        depth_image = np.asanyarray(aligned_depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())

        blue_image = color_image.copy()
        blue_image[:, :, 0] = 0
        blue_image[:, :, 1] = 0

        grey_color = 153
        depth_image_3d = np.dstack((depth_image, depth_image, depth_image))
        bg_removed = np.where((depth_image_3d < clipping_distance_min) | (depth_image_3d > clipping_distance_max) | (
                    depth_image_3d <= 0), grey_color, blue_image)

        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
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
            M = cv2.getPerspectiveTransform(npcorners, dst)
            grid = cv2.warpPerspective(bg_removed, M, (maxWidth, maxHeight))
            cv2.imshow('grid', grid)
            img_resized = cv2.resize(grid, (1920, 1080), interpolation=cv2.INTER_AREA)
            hsv = cv2.cvtColor(img_resized, cv2.COLOR_BGR2HSV)
            lower_red = np.array([0, 50, 50])
            upper_red = np.array([10, 255, 255])
            mask1 = cv2.inRange(hsv, lower_red, upper_red)
            lower_red = np.array([170, 50, 50])
            upper_red = np.array([180, 255, 255])
            mask2 = cv2.inRange(hsv, lower_red, upper_red)
            mask = mask1 + mask2
            contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            min_contour_area = 2000  # Puedes ajustar este valor según sea necesario es un filtro de contorno por area
            filtered_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_contour_area]

            if len(filtered_contours) > 0:
                largest_contour = max(filtered_contours, key=cv2.contourArea)
                M = cv2.moments(largest_contour)
                if M['m00'] == 0:
                    cx = 0
                else:
                    cx = int(M['m10'] / M['m00']) + 1920  # el 1920 indica que se corren las cordenadas para una pantalla estendida
                    cy = int(M['m01'] / M['m00'])

                print("El centro del área roja está en las coordenadas ({}, {})".format(cx, cy))
                # Mueve el cursor a la posición deseada
                pyautogui.moveTo(cx, cy)
                # # Hace clic en la posición actual del cursor
                # pyautogui.click()
                import time
                time.sleep(0.4)

        cv2.polylines(color_image, [np.array(corners)], isClosed=True, color=(0, 255, 0), thickness=2)

        cv2.putText(color_image, "Iniciar touch", (button_x, button_y + 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5,
                    (255, 0, 0), 2, cv2.LINE_AA)
        cv2.rectangle(color_image, (button_x, button_y), (button_x + button_w, button_y + button_h), (255, 0, 0), 2)
        
        cv2.putText(color_image, "Finalizar", (finish_button_x, finish_button_y + 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5,
                    (255, 0, 0), 2, cv2.LINE_AA)
        cv2.rectangle(color_image, (finish_button_x, finish_button_y), (finish_button_x + finish_button_w, finish_button_y + finish_button_h), (255, 0, 0), 2)

        cv2.imshow('Camera', color_image)

        key = cv2.waitKey(1)

        if key & 0xFF == ord('q') or key == 27:
            print(corners)
            cv2.destroyAllWindows()
            break

finally:
    pipeline.stop()
