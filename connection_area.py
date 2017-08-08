# -*- coidng=utf-8 -*-
import cv2
import numpy as np


class Stack:

    def __init__(self):
        self.items = []

    def isEmpty(self):
        return len(self.items) == 0

    def push(self, item):
        self.items.append(item)

    def pop(self):
        return self.items.pop()

    def peek(self):
        return self.items[-1]

    def size(self):
        return len(self.items)


def find_grey_connected_area(img, color_diff=25, backgroud=255, size=5):
    """
    find threshold image connected area
    """
    label_img = np.zeros_like(img)
    label = 0
    label_points = {}

    rows, cols = img.shape
    for row in range(0, rows):
        for col in range(0, cols):
            if label_img[row, col]:
                continue
            if img[row, col] == backgroud:
                continue

            label += 1
            stack = Stack()
            stack.push([row, col])
            while not stack.isEmpty():
                cur_row, cur_col = stack.pop()
                label_img[cur_row, cur_col] = label
                label_points.setdefault(label, []).append([cur_row, cur_col])
                points = find_near_color_points(img, cur_row, cur_col,
                        backgroud=backgroud, color_diff=color_diff, size=size)
                for point in points:
                    if point[0] == row and point[1] == col:
                        continue
                    if label_img[point[0], point[1]]:
                        continue
                    stack.push(point)

    return label_img, label_points


def find_near_color_points(image, row, col, backgroud, color_diff, size):
    height, width = image.shape
    border = size / 2
    cur_value = image[row, col]

    start_row = row - border
    if start_row < 0:
        start_row = 0
    end_row = row + border
    if end_row >= height:
        end_row = height -1

    start_col = col - border
    if start_col < 0:
        start_col = 0
    end_col = col + border
    if end_col >= width:
        end_col = width - 1

    points = []

    def mul_inserct(list_a, list_b):
        result = []
        for a in list_a:
            for b in list_b:
                result.append([a, b])
        return result

    near_points = mul_inserct(range(start_row, end_row+1),
                                range(start_col, end_col+1))
    for r, c in near_points:
        if image[r, c] == backgroud:
            continue
        if r == row and c == col:
            continue
        diff = abs(int(image[r, c]) - int(cur_value))
        if diff < color_diff:
            points.append([r, c])

    return points
