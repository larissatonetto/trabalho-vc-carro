import numpy as np
import cv2 as cv
from datetime import datetime

# change the resolution and fps to match your device
fps = 30.0
height = 240.0
width = 320.0

x_start = int(width / 2 - width / 4)
x_end = int(width / 2 + width / 4)
y_start = int(height / 2 - height / 4)
y_end = int(height / 2 + height / 4)

def get_dominant_color(img):
    img_hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    arr = np.asarray(img_hsv).astype(np.uint8)

    # Cria um box de tamanho largura/2 * altura/2 no meio da tela
    x_start = int(width / 2 - width / 4)
    x_end = int(width / 2 + width / 4)
    y_start = int(height / 2 - height / 4)
    y_end = int(height / 2 + height / 4)

    arr[y_start, x_start:x_end] = (0, 0, 0)
    arr[y_end, x_start:x_end] = (0, 0, 0)
    arr[y_start:y_end, x_start] = (0, 0, 0)
    arr[y_start:y_end, x_end] = (0, 0, 0)

    hue = arr[y_start:y_end, x_start:x_end, 0]
    sat = arr[y_start:y_end, x_start:x_end, 1]
    val = arr[y_start:y_end, x_start:x_end, 2]

    sum_black = np.sum(val < 30)
    sum_red = np.sum((hue >= 0) & (hue < 60))
    sum_green = np.sum((hue >= 60) & (hue < 120))
    sum_blue = np.sum((hue >= 120) & (hue < 180))

    colors = {0: "black", 1: "red", 2: "green", 3: "blue"}
    color_sums = [
        np.sum(val < 60),  # Preto
        np.sum((hue >= 0) & (hue < 60)),  # Vermelho
        np.sum((hue >= 60) & (hue < 120)),  # Verde
        np.sum((hue >= 120) & (hue <= 180)),  # Azul
    ]

    return (cv.cvtColor(arr, cv.COLOR_HSV2BGR), colors[np.argmax(color_sums)])


def show_color(arr, color):
    # bgr_arr = np.asarray(img).astype(np.uint8)
    img_hsv = cv.cvtColor(arr, cv.COLOR_BGR2HSV)

    # bgr_arr[y_start:y_end, x_start:x_end] = (255, 255, 255)
    # bgr_arr[y_start, x_start:x_end] = (0, 0, 0)
    # bgr_arr[y_end, x_start:x_end] = (0, 0, 0)
    # bgr_arr[y_start:y_end, x_start] = (0, 0, 0)
    # bgr_arr[y_start:y_end, x_end] = (0, 0, 0)

    hue = img_hsv[:,:,0]
    sat = img_hsv[:,:,1]
    val = img_hsv[:,:,2]

    new_arr = arr.copy()

    if color == "black":
        new_arr[val > 50] = (255, 255, 255)
    elif color == "red":
        new_arr[(hue >= 60) | (val <= 60)] = (255, 255, 255)
    elif color == "blue":
        new_arr[(hue < 60) | (hue >= 120) | (val <= 60)] = (255, 255, 255)
    else:
        new_arr[(hue < 120) | (val <= 60)] = (255, 255, 255)

    return new_arr


def process_image(image):
    arr = np.asarray(image).astype(np.uint8)
    new_img = show_color(arr[y_start:y_end, x_start:x_end], "green")
    # print(color)
    cv.imshow("frame", new_img)

def process_image_debug(image):
    arr = np.asarray(image).astype(np.uint8)
    img_black = show_color(arr[y_start:y_end, x_start:x_end], "black")
    img_red = show_color(arr[y_start:y_end, x_start:x_end], "red")
    img_green = show_color(arr[y_start:y_end, x_start:x_end], "green")
    img_blue = show_color(arr[y_start:y_end, x_start:x_end], "blue")

    cv.imshow("black", img_black)
    cv.imshow("red", img_red)
    cv.imshow("green", img_green)
    cv.imshow("blue", img_blue)
    cv.imshow("original", image)

# Change the camera id
cap = cv.VideoCapture(0)

if not cap.isOpened():
    print("Cannot open camera")
    exit()

cap.set(cv.CAP_PROP_FPS, fps)
cap.set(cv.CAP_PROP_FRAME_HEIGHT, height)
cap.set(cv.CAP_PROP_FRAME_WIDTH, width)


if (
    (cap.get(cv.CAP_PROP_FRAME_WIDTH) != width)
    or (height != cap.get(cv.CAP_PROP_FRAME_HEIGHT))
    or (fps != cap.get(cv.CAP_PROP_FPS))
):
    print("ERRO na configuração da câmera.")
    print(f"Width: {cap.get(cv.CAP_PROP_FRAME_WIDTH)}")
    print(f"Height: {cap.get(cv.CAP_PROP_FRAME_HEIGHT)}")
    print(f"FPS: {cap.get(cv.CAP_PROP_FPS)}")
else:
    print("Configuração de câmera OK.")


while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    # if frame is read correctly ret is True
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break

    # process_image(frame)
    process_image_debug(frame)

    print(datetime.utcnow().strftime("%F %T.%f")[:-3])

    if cv.waitKey(1) == ord("q"):
        break

# When everything done, release the capture
cap.release()
cv.destroyAllWindows()
