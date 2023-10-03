import numpy as np
import cv2 as cv
from datetime import datetime
import serial

# change the resolution and fps to match your device
fps = 10.0
height = 240.0
width = 320.0

x_start = int(width / 2 - width / 4)
x_end = int(width / 2 + width / 4)
y_start = int(height / 2 - height / 4)
y_end = int(height / 2 + height / 4)

v = 35
s = 35


def getDominantColor(arr):
    img_hsv = cv.cvtColor(arr, cv.COLOR_BGR2HSV)

    arr[y_start, x_start:x_end] = (0, 0, 0)
    arr[y_end, x_start:x_end] = (0, 0, 0)
    arr[y_start:y_end, x_start] = (0, 0, 0)
    arr[y_start:y_end, x_end] = (0, 0, 0)

    b, g, r = cv.split(arr[y_start:y_end, x_start:x_end])
    hue, sat, val = cv.split(img_hsv[y_start:y_end, x_start:x_end])

    keys = {0: "s", 1: "a", 2: "d", 3: "w"}
    color_sums = [
        np.sum(val < v),  # Preto
        np.sum(np.sum((r > b) & (r > g) & (val > v) & (sat >= s))),  # Vermelho
        np.sum(np.sum((g > r) & (g > b) & (val > v) & (sat >= s))),  # Verde
        np.sum(np.sum((b > r) & (b > g) & (val > v) & (sat >= s))),  # Azul
    ]

    return (arr, keys[np.argmax(color_sums)])


def showColor(arr, color):
    img_hsv = cv.cvtColor(arr, cv.COLOR_BGR2HSV)
    new_arr = arr.copy()

    b, g, r = cv.split(arr)
    hue, sat, val = cv.split(img_hsv)

    if color == "Preto":
        new_arr[val >= v] = (255, 255, 255)
    elif color == "Vermelho":
        new_arr[(r < b) | (r < g) | (val < v) | (sat < s)] = (255, 255, 255)
    elif color == "Verde":
        new_arr[(g < r) | (g < b) | (val < v) | (sat < s)] = (255, 255, 255)
    else:
        new_arr[(b < r) | (b < g) | (val < v) | (sat < s)] = (255, 255, 255)

    return new_arr


def processImage(image):
    arr = np.asarray(image).astype(np.uint8)
    new_img, key = getDominantColor(arr)
    cv.imshow("frame", new_img)
    return key


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

ser = serial.Serial("/dev/ttyACM0", 9600)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    # if frame is read correctly ret is True
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break

    key = processImage(frame)
    processImageDebug(frame)
    if key == ("w"):
        #ser.write(b"w")
        print("w")
    elif key == ("a"):
        # ser.write(b"a")
        print("a")
    elif key == ("s"):
        # ser.write(b"s")
        print("s")
    else:
        # ser.write(b"d")
        print("d")

    print(datetime.utcnow().strftime("%F %T.%f")[:-3])

    if cv.waitKey(1) == ord("q"):
        break

# When everything done, release the capture
cap.release()
ser.close()
cv.destroyAllWindows()

