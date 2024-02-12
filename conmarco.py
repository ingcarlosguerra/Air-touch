# First import the library
import pyrealsense2 as rs
# Import Numpy for easy array manipulation
import numpy as np
# Import OpenCV for easy image rendering
import cv2
# Import Pyautogui to gain access to click 
import pyautogui

# Create a pipeline
pipeline = rs.pipeline()
config = rs.config()
# Get device product line for setting a supporting resolution
pipeline_wrapper = rs.pipeline_wrapper(pipeline)
pipeline_profile = config.resolve(pipeline_wrapper)
device = pipeline_profile.get_device()
device_product_line = str(device.get_info(rs.camera_info.product_line))
found_rgb = False
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

# Start streaming
profile = pipeline.start(config)
# Getting the depth sensor's depth scale (see rs-align example for explanation)
depth_sensor = profile.get_device().first_depth_sensor()
depth_scale = depth_sensor.get_depth_scale()
print("Depth Scale is: " , depth_scale)

# We will be removing the background of objects more than
#  clipping_distance_in_meters meters away
clipping_distance_in_meters_max = 1.2#1 meter
clipping_distance_in_meters_min = 1.1#1 meter

clipping_distance_max = clipping_distance_in_meters_max / depth_scale
clipping_distance_min = clipping_distance_in_meters_min / depth_scale
print("Print min:", clipping_distance_min,  "printo max:", clipping_distance_max)
align_to = rs.stream.color
align = rs.align(align_to)

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

# Configurar la ventana de la cámara y establecer la función de callback del mouse
cv2.namedWindow('Camera')
cv2.setMouseCallback('Camera', on_mouse)
cv2.namedWindow('grid')
# Streaming loop
try:
    while True:
        # Get frameset of color and depth
        frames = pipeline.wait_for_frames()
        # frames.get_depth_frame() is a 640x360 depth image

        # Align the depth frame to color frame
        aligned_frames = align.process(frames)

        # Get aligned frames
        aligned_depth_frame = aligned_frames.get_depth_frame() # aligned_depth_frame is a 640x480 depth image
        color_frame = aligned_frames.get_color_frame()

        # Validate that both frames are valid
        if not aligned_depth_frame or not color_frame:
            continue

        depth_image = np.asanyarray(aligned_depth_frame.get_data())
        # depth_image=  cv2.rotate(depth_image, cv2.ROTATE_180)

        color_image = np.asanyarray(color_frame.get_data())
        blue_image = color_image.copy()
        blue_image[:, :, 0] = 0
        blue_image[:, :, 1] = 0
        # Seleccionar solo el canal azul
        # Remove background - Set pixels further than clipping_distance to grey
        grey_color = 153
        depth_image_3d = np.dstack((depth_image,depth_image,depth_image)) #depth image is 1 channel, color is 3 channels
        # bg_removed = np.where((depth_image_3d > clipping_distance_max) | (depth_image_3d <= 0), grey_color, color_image)
        bg_removed = np.where((depth_image_3d < clipping_distance_min) | (depth_image_3d > clipping_distance_max) | (depth_image_3d <= 0), grey_color,blue_image)
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
        cv2.polylines(color_image, [np.array(corners)], isClosed=True, color=(0, 0, 255), thickness=2)
        corner1=(corners[0][0],corners[0][1])
        corner2=(corners[1][0],corners[1][1])
        corner3=(corners[2][0],corners[2][1])
        corner4=(corners[3][0],corners[3][1])

        npcorners = np.array([corner1,corner2,corner3,corner4], dtype="float32")
                # the width of the new frame
        widthA = np.sqrt(((corner3[0] - corner4[0]) ** 2) + ((corner3[1] - corner4[1]) ** 2))
        widthB = np.sqrt(((corner2[0] - corner1[0]) ** 2) + ((corner2[1] - corner1[1]) ** 2))

        #the height of the new frame
        heightA = np.sqrt(((corner2[0] - corner3[0]) ** 2) + ((corner2[1] - corner3[1]) ** 2))
        heightB = np.sqrt(((corner1[0] - corner4[0]) ** 2) + ((corner1[1] - corner4[1]) ** 2))

        #final dimensions
        maxWidth = max(int(widthA), int(widthB))
        maxHeight = max(int(heightA), int(heightB))

        # construct our destination points which will be used to
        # map the screen to a top-down, "birds eye" view
        dst = np.array([
            [0, 0],
            [maxWidth - 1, 0],
            [maxWidth - 1, maxHeight - 1],
            [0, maxHeight - 1]], dtype = "float32")

        M = cv2.getPerspectiveTransform(npcorners, dst)
        grid = cv2.warpPerspective(bg_removed, M, (maxWidth, maxHeight))
        # grid es la variable que me permite solo analizar el area deseada

        img_resized = cv2.resize(grid, (1920, 1080), interpolation = cv2.INTER_AREA)
        hsv = cv2.cvtColor(img_resized, cv2.COLOR_BGR2HSV)
        # Define el rango de colores rojos en HSV
        lower_red = np.array([0, 50, 50])
        upper_red = np.array([10, 255, 255])
        mask1 = cv2.inRange(hsv, lower_red, upper_red)
        lower_red = np.array([170, 50, 50])
        upper_red = np.array([180, 255, 255])
        mask2 = cv2.inRange(hsv, lower_red, upper_red)
        # Combina las dos máscaras
        mask = mask1 + mask2
        # Encuentra los contornos de la máscara
        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

                # Filtra los contornos por área mínima deseada
        min_contour_area = 2000  # Puedes ajustar este valor según sea necesario

        filtered_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_contour_area]

        if len(filtered_contours) > 0:
                    # Encuentra el contorno más grande
            largest_contour = max(filtered_contours, key=cv2.contourArea)

            # Encuentra el centro del contorno más grande
            M = cv2.moments(largest_contour)
            if M['m00'] == 0:
                # Manejar el error de división por cero aquí
                cx = 0
            else:
                cx = int(M['m10'] / M['m00'])
                cy = int(M['m01'] / M['m00'])

         
            
            # Imprime las coordenadas del centro del área
            print("El centro del área roja está en las coordenadas ({}, {})".format(cx, cy))
            # Mueve el cursor a la posición deseada
            # pyautogui.moveTo(cx, cy)
            # # Hace clic en la posición actual del cursor
            # pyautogui.click()
            import time
            time.sleep(0.35) 

        cv2.imshow('Camera', color_image)
        cv2.imshow('grid', grid)
        key = cv2.waitKey(1)
        # Press esc or 'q' to close the image window
        if key & 0xFF == ord('q') or key == 27:
            print(corners)
            cv2.destroyAllWindows()
            break
    
finally:
    pipeline.stop()