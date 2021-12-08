import sys
import time

import matplotlib.pyplot as plt
import cv2
import numpy as np
from shapely.geometry import Point, Polygon

from numpy import ndarray

figsize = (10, 10)
on = False
# on = False
font = cv2.FONT_HERSHEY_SIMPLEX
bottomLeftCornerOfText = (10, 500)
fontScale = 2
fontColor = (50, 255, 255)
lineType = 2


def gaussianBlur_and_canny(image):
    gs_image = cv2.GaussianBlur(image, (15, 15), 2)
    canny_image = cv2.Canny(gs_image, 160, 200)
    return canny_image


def cut_image_to_center(image):
    image_height = image.shape[0]
    image_w = image.shape[0]
    w = (1 / 5) * image_w
    w2 = (8 / 5) * image_w
    center_of_the_car = np.array(
        [[(int(w), image_height - 100), (int(w), 550), (int(w2), 550), (int(w2), image_height - 100)]])
    #center_of_the_car = np.array([[(400, 450), (200, image_height - 100), (1500, image_height - 100)]])

    mask = np.zeros_like(image)
    cv2.fillConvexPoly(mask, center_of_the_car, 1)
    if on:
        plt.figure(figsize=figsize)
        plt.imshow(mask)
        plt.show()
    center_of_the_image = cv2.bitwise_and(image, mask)
    if on:
        plt.figure(figsize=figsize)
        plt.imshow(center_of_the_image)
        plt.show()
    return center_of_the_image


def check_line_transfer(image, lines, text):
    image_height = image.shape[0]
    #center_of_the_car_line_check = np.array([[(475, image_height - 100),  (550, 200), (675, image_height - 100)]])
    center_of_the_car_line_check = np.array([[(485, image_height - 100), (550, 200), (675, image_height - 100)]])
    polygon = Polygon(center_of_the_car_line_check[0])
    for line in lines:
        if text == "":
            x0, y0, x1, y1 = line[0]
            parameters = np.polyfit((x0, x1), (y0, y1), 1)
            point1 = Point(x0, y0)
            point2 = Point(x1, y1)
            if polygon.contains(point1) or polygon.contains(point2):
                direction = parameters[0]
                if direction < 0:
                    text = "Changeing lane to the left"
                else:
                    text = "Changeing lane to the right"

                print(text)
                break;
    return text


def get_line_cord(image_heigt, line):
    m, b = line
    y1 = image_heigt
    y2 = int(y1 * (3.5 / 5))
    x1 = int((y1 - b) / m)
    x2 = int((y2 - b) / m)
    return np.array([x1, y1, x2, y2])


def get_left_and_rigt_lines(image, lines):
    left_line = []
    right_line = []
    for line in lines:
        x1, y1, x2, y2 = line.reshape(4)
        param = np.polyfit((x1, x2), (y1, y2), 1)
        incline = param[0]
        cutin = param[1]
        if (not -0.5 <= incline <= 0.5) or (incline < -1 or incline > 1):
            if incline < 0:
                left_line.append((incline, cutin))
            else:
                right_line.append((incline, cutin))

    if left_line:
        left_avg = np.average(left_line, axis=0)
        left_line = get_line_cord(image.shape[0] - 100, left_avg)

    if right_line:
        right_avg = np.average(right_line, axis=0)
        right_line = get_line_cord(image.shape[0] - 100, right_avg)

    return np.array([left_line, right_line], dtype=object)


def draw_lines(image, lines):
    res = image.copy()
    for line in lines:
        if line is not None:
            x0, y0, x1, y1 = line
            res = cv2.line(res, (int(x0), int(y0)), (int(x1), int(y1)), (255, 0, 0), 15)

    return res


def get_avrage_line(line):
    if line:
        return np.average(line, axis=0)


cap = cv2.VideoCapture("carRide_Trim.mp4")
i = 0
ret, frame = cap.read()
text = ""
total_left_lines = []
total_right_lines = []

while cap.isOpened() and ret:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    image2 = gaussianBlur_and_canny(gray)

    image = cut_image_to_center(image2)

    lines = cv2.HoughLinesP(image, 1, np.pi / 90, threshold=60, minLineLength=50, maxLineGap=200)

    if lines is not None:
        if i == 60:
            text = ""
            i = 0
        else:
            i += 1
        text = check_line_transfer(frame, lines, text)
        lines_to_save = get_left_and_rigt_lines(frame, lines)

        if len(lines_to_save[0]) > 0:
            total_left_lines.append(lines_to_save[0])
            left_line_to_draw = lines_to_save[0]
        else:
            left_line_to_draw = get_avrage_line(total_left_lines)

        if len(lines_to_save[1]) > 0:
            total_right_lines.append(lines_to_save[1])
            right_line_to_draw = lines_to_save[1]
        else:
            right_line_to_draw = get_avrage_line(total_right_lines)


    else:
        left_line_to_draw = get_avrage_line(total_left_lines)
        right_line_to_draw = get_avrage_line(total_right_lines)

    # time.sleep(0.005)
    lines_to_draw = [left_line_to_draw, right_line_to_draw]
    if text == "":
        frame = draw_lines(frame, lines_to_draw)
    cv2.putText(frame, text, (100, 100), font, fontScale, fontColor, 2, lineType)
    cv2.imshow('frame', frame)
    ret, frame = cap.read()

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
