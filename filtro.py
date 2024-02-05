import pyrealsense2 as rs
import numpy as np
import cv2
import pyautogui

pipeline = rs.pipeline()
config = rs.config()

found_rgb = False
for s in pipeline_profile.get_device().sensors:
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

pipeline.start(config)
profile = pipeline.start(config)

depth_sensor = profile.get_device().first_depth_sensor()
depth_scale = depth_sensor.get_depth_scale()
print("Depth Scale is: ", depth_scale)

clipping_distance_in_meters_max = 1.1  # 1 meter
clipping_distance_in_meters_min = 1.0  # 1 meter

clipping_distance_max = clipping_distance_in_meters_max / depth_scale
clipping_distance_min = clipping_distance_in_meters_min / depth_scale
print("Print min:", clipping_distance_min, "printo max:", clipping_distance_max)

align_to = rs.stream.color
align = rs.align(align_to)

try:
    while True:
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
        bg_removed = np.where(
            (depth_image_3d < clipping_distance_min) | (depth_image_3d > clipping_distance_max) | (
                        depth_image_3d <= 0), grey_color, blue_image)

        img_resized = cv2.resize(bg_removed, (1920, 1080), interpolation=cv2.INTER_AREA)
        hsv = cv2.cvtColor(img_resized, cv2.COLOR_BGR2HSV)
        lower_red = np.array([0, 50, 50])
        upper_red = np.array([10, 255, 255])
        mask1 = cv2.inRange(hsv, lower_red, upper_red)
        lower_red = np.array([170, 50, 50])
        upper_red = np.array([180, 255, 255])
        mask2 = cv2.inRange(hsv, lower_red, upper_red)
        mask = mask1 + mask2
        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # Filtra contornos basados en el área (ajusta el valor según tus necesidades)
        area_umbral = 500
        contornos_filtrados = [contorno for contorno in contours if cv2.contourArea(contorno) > area_umbral]

        # Dibuja los contornos filtrados en la imagen original
        contornos_img = np.zeros_like(bg_removed)
        cv2.drawContours(contornos_img, contornos_filtrados, -1, (255, 255, 255), 2)

        # Aplica la máscara de contornos a la imagen original
        bg_removed_filtered = cv2.bitwise_and(bg_removed, bg_removed, mask=cv2.bitwise_not(contornos_img))

        # Render images
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
        images = np.hstack((bg_removed_filtered, depth_colormap))

        cv2.namedWindow('Align Example', cv2.WINDOW_NORMAL)
        cv2.imshow('Align Example', images)
        key = cv2.waitKey(1)

        if key & 0xFF == ord('q') or key == 27:
            cv2.destroyAllWindows()
            break
finally:
    pipeline.stop()
