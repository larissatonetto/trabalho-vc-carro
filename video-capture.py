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


def getDominantColor(arr):
    img_hsv = cv.cvtColor(arr, cv.COLOR_BGR2HSV)

    arr[y_start, x_start:x_end] = (0, 0, 0)
    arr[y_end, x_start:x_end] = (0, 0, 0)
    arr[y_start:y_end, x_start] = (0, 0, 0)
    arr[y_start:y_end, x_end] = (0, 0, 0)

    b, g, r = np.split(arr[y_start:y_end, x_start:x_end], 3, axis=2)
    hue, sat, val = np.split(img_hsv[y_start:y_end, x_start:x_end], 3, axis=2)

    commands = {0: "s", 1: "a", 2: "d", 3: "w"}
    color_sums = [
        np.sum(val < 50),  # Preto
        np.sum(np.sum((r > b) & (r > g) & (val > 50) & (sat > 50))),  # Vermelho
        np.sum(np.sum((g > r) & (g > b) & (val > 50) & (sat > 50))),  # Verde
        np.sum(np.sum((b > r) & (b > g) & (val > 50) & (sat > 50))),  # Azul
    ]

    return (arr, commands[np.argmax(color_sums)])


def showColor(arr, color):
    img_hsv = cv.cvtColor(arr, cv.COLOR_BGR2HSV)
    new_arr = arr.copy()

    b, g, r = np.split(arr[y_start:y_end, x_start:x_end], 3, axis=2)
    hue, sat, val = np.split(img_hsv[y_start:y_end, x_start:x_end], 3, axis=2)

    if color == "Preto":
        new_arr[val > 50] = (255, 255, 255)
    elif color == "Vermelho":
        new_arr[(r < b) | (r < g) | (val < 50) | (sat < 50)] = (255, 255, 255)
    elif color == "Verde":
        new_arr[(g < r) | (g < b) | (val < 50) | (sat < 50)] = (255, 255, 255)
    else:
        new_arr[(b < r) | (b < g) | (val < 50) | (sat < 50)] = (255, 255, 255)

    return new_arr


def processImage(image):
    arr = np.asarray(image).astype(np.uint8)
    new_img, command = getDominantColor(arr)
    cv.imshow("frame", new_img)
    return command


def processImageDebug(image):
    arr = np.asarray(image).astype(np.uint8)
    img_black = showColor(arr, "Preto")
    img_red = showColor(arr, "Vermelho")
    img_green = showColor(arr, "Verde")
    img_blue = showColor(arr, "Azul")

    cv.imshow("Preto", img_black)
    cv.imshow("Vermelho", img_red)
    cv.imshow("Verde", img_green)
    cv.imshow("Azul", img_blue)
    cv.imshow("Original", image)


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

    comm = processImage(frame)
    print(comm)
    # processImageDebug(frame)

    print(datetime.utcnow().strftime("%F %T.%f")[:-3])

    if cv.waitKey(1) == ord("q"):
        break

# When everything done, release the capture
cap.release()
cv.destroyAllWindows()
