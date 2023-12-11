import numpy as np
import time
import cv2 as cv
from datetime import datetime
import serial
import matplotlib.pyplot as plt

# change the resolution and fps to match your device
fps = 30.0
height = 240.0
width = 320.0

x_start = int(width / 2 - width / 4)
x_end = int(width / 2 + width / 4)
y_start = int(height / 2 - height / 4)
y_end = int(height / 2 + height / 4)

V = 35
s = 35


def get_mask_pista(img):
    # print(img.shape)
    img_hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    b, g, r = cv.split(img)
    h, s, v = cv.split(img_hsv)

    new_img = np.zeros_like(img)

    new_img[(s > 50) & (b > g) & (b > r)] = (255, 255, 255)

    return new_img


def get_dom_color(img):
    img_hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)

    img[y_start, x_start:x_end] = (0, 0, 0)
    img[y_end, x_start:x_end] = (0, 0, 0)
    img[y_start:y_end, x_start] = (0, 0, 0)
    img[y_start:y_end, x_end] = (0, 0, 0)

    b, g, r = cv.split(img[y_start:y_end, x_start:x_end])
    hue, sat, val = cv.split(img_hsv[y_start:y_end, x_start:x_end])

    keys = {0: "s", 1: "a", 2: "d", 3: "w"}
    color_sums = [
        np.sum(val < v),  # Preto
        np.sum(np.sum((r > b) & (r > g) & (val > v) & (sat >= s))),  # Vermelho
        np.sum(np.sum((g > r) & (g > b) & (val > v) & (sat >= s))),  # Verde
        np.sum(np.sum((b > r) & (b > g) & (val > v) & (sat >= s))),  # Azul
    ]

    return (img, keys[np.argmax(color_sums)])


# Detecta círculos
def detecta_placa(img):
    stop = False
    limit_x = int(2 * width / 3)
    # bimg = cv.medianBlur(img, 5)
    img_hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)

    h, s, v = cv.split(img_hsv)
    b, g, r = cv.split(img)

    new_img = np.zeros_like(img)
    new_img[(b > g) & (b > r) & (v > V) & (s > 50)] = (255, 255, 255)

    cimg = cv.cvtColor(new_img, cv.COLOR_BGR2GRAY)

    circles = cv.HoughCircles(
        cimg[0 : int(height / 3), int(2 * width / 3) : int(width)],
        cv.HOUGH_GRADIENT,
        1,
        5,
        param1=50,
        param2=15,
        minRadius=5,
        maxRadius=20,
    )

    if circles is not None:
        circles = np.uint16(np.around(circles))
        # print("len = ", len(circles))
        # print(circles)

        if circles[0][0][2] < 4:
            cv.circle(
                new_img,
                (circles[0][0][0] + limit_x, circles[0][0][1]),
                circles[0][0][2],
                (255, 0, 0),
                2,
            )
        elif circles[0][0][2] < 10:
            cv.circle(
                new_img,
                (circles[0][0][0] + limit_x, circles[0][0][1]),
                circles[0][0][2],
                (0, 255, 255),
                2,
            )
        else:
            cv.circle(
                new_img,
                (circles[0][0][0] + limit_x, circles[0][0][1]),
                circles[0][0][2],
                (0, 0, 255),
                2,
            )

            stop = True

        cv.circle(
            new_img, (circles[0][0][0] + limit_x, circles[0][0][1]), 2, (0, 0, 255), 3
        )

    return [new_img, stop]


def canny(img):
    if len(img.shape) == 3:
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    else:
        gray = img

    blur = cv.GaussianBlur(gray, (3, 3), 0)
    canny = cv.Canny(blur, 20, 80)

    return canny


def region_of_interest(img):
    height, width = img.shape

    triangle = np.array(
        [(0, height), (width, height), (width / 2, int(height / 4))], dtype=np.int32
    )

    mask = np.zeros_like(img, dtype=np.uint8)
    cv.fillPoly(mask, [triangle], (255, 255, 255))

    masked_image = cv.bitwise_and(img, mask)

    return masked_image


# Debug para ver as cores detectadas separadamente
def show_color(img, color):
    img_hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    new_img = img.copy()

    b, g, r = cv.split(img)
    hue, sat, val = cv.split(img_hsv)

    if color == "Preto":
        new_img[val >= v] = (255, 255, 255)
    elif color == "Vermelho":
        new_img[(r < b) | (r < g) | (val < v) | (sat < s)] = (255, 255, 255)
    elif color == "Verde":
        new_img[(g < r) | (g < b) | (val < v) | (sat < s)] = (255, 255, 255)
    else:
        new_img[(b < r) | (b < g) | (val < v) | (sat < s)] = (255, 255, 255)

    return new_img


# Encontra os elementos da pista e retorna o comando para movimentar o carro
def processImage(image):
    arr = np.asarray(image).astype(np.uint8)
    new_img, key = get_dom_color(arr)
    cv.imshow("frame", new_img)
    return key


# Encontra os elementos da pista e abre uma janela para cada um
def processImageDebug(image):
    arr = np.asarray(image).astype(np.uint8)
    # img_black = show_color(arr, "Preto")
    # img_red = show_color(arr, "Vermelho")
    # img_green = show_color(arr, "Verde")
    # img_blue = show_color(arr, "Azul")
    img_circles = detecta_placa(arr)
    # img_lane = detectLane(arr)

    cv.imshow("Circulo", img_circles)
    # cv.imshow("Preto", img_black)
    # cv.imshow("Vermelho", img_red)
    # cv.imshow("Verde", img_green)
    # cv.imshow("Azul", img_blue)
    # cv.imshow("Lane", img_lane)
    # cv.imshow("Original", image)


# ser = serial.Serial("/dev/ttyACM0", 9600)


def detecta_linha(edges, frame):
    lines = cv.HoughLines(edges, 1, np.pi / 180, 100)
    # print(lines)
    turnLeft = False
    turnRight = False

    if len(frame.shape) != 3:
        new_frame = np.stack([frame] * 3, axis=-1)
    else:
        new_frame = frame.copy()

    # print(new_frame.shape)

    if lines is not None:
        for line in lines:
            rho, theta = line[0]
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho
            x1 = int(x0 + 1000 * (-b))
            y1 = int(y0 + 1000 * (a))
            x2 = int(x0 - 1000 * (-b))
            y2 = int(y0 - 1000 * (a))

            if theta >= 1.9:
                cv.line(new_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                turnLeft = True

            if theta <= 1.45:
                cv.line(new_frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                turnRight = True

    return [new_frame, turnLeft, turnRight]


def dilation(img):
    kernel = np.ones((2, 2), np.uint8)
    d = cv.dilate(img, kernel)

    return d


def teste_imagem():
    imagem = cv.imread("esquerda.png")

    mask = get_mask_pista(imagem)
    edges = canny(mask)  # Isola a pista antes de detectar as bordas
    cropped_image = region_of_interest(edges)
    edges2 = canny(cropped_image)

    f1 = detecta_linha(edges, imagem)
    f2 = detecta_linha(edges2, cropped_image)

    cv.imshow("result", f2[0])
    cv.imshow("imagem", f1[0])
    cv.imshow("dilation", dilation(f2[0]))

    # plt.subplot(122), plt.imshow(edges, cmap="gray")
    # plt.title("Edge Image 1"), plt.xticks([]), plt.yticks([])
    # plt.show()

    # plt.subplot(122), plt.imshow(edges2, cmap="gray")
    # plt.title("Edge Image 2"), plt.xticks([]), plt.yticks([])
    # plt.show()

    cv.waitKey(0)


def teste_video():
    cap = cv.VideoCapture("output.avi")
    parou = False

    while cap.isOpened():
        _, frame = cap.read()

        arr = np.asarray(frame).astype(np.uint8)

        mask = get_mask_pista(arr)
        edges = canny(mask)  # Isola a pista antes de detectar as bordas
        cropped_image = region_of_interest(edges)
        edges2 = canny(cropped_image)

        f1 = detecta_linha(edges, arr)
        f2 = detecta_linha(edges2, dilation(cropped_image))
        circ = detecta_placa(arr)

        cv.imshow("imagem", f1[0])
        cv.imshow("result", f2[0])
        cv.imshow("placa", circ[0])

        if (circ[1] == 1) & (not parou):
            print("parar")
            time.sleep(5)
            parou = True
        if f2[1] == 1:
            print("a")
        elif f2[2] == 1:
            print("d")
        else:
            print("a")

        k = cv.waitKey(20) & 0xFF
        if k == 27:
            break


teste_video()

if cap is not None:
    cap.release()
# ser.close()
cv.destroyAllWindows()
