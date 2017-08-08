# -*-coding:utf-8 -*-

import os
import cv2
import glob
import time
import numpy as np

from math import atan2
from itertools import combinations
from matplotlib import pyplot as plt
from connection_area import find_grey_connected_area



def standard_size(img, path):
    # paths = glob.glob(r'/home/yyz/number-plate/grey-samples/others/*.jpg')
    # for path in paths:
    #     print path
    #     img = cv2.imread(path)
    image = np.empty([200,200,3], dtype=int)
    image.fill(255)
    rows, cols = img.shape[0:2]
    col=0
    row=0
    while col<cols:
        y = img[:,col]
        if np.count_nonzero(y!=255)>5:
            start_col = col
            end_col = cols - 1
            while end_col>start_col:
                y = img[:,end_col]
                if np.count_nonzero(y!=255)>5:
                    break
                end_col-=1
            break
        col+=1
    while row<rows:
        x = img[row,:]
        if np.count_nonzero(x!=255)>5:
            start_row = row
            end_row = rows - 1
            while end_row>start_row:
                x = img[end_row,:]
                if np.count_nonzero(x!=255)>5:
                    break
                end_row-=1
            break
        row+=1

    dif_y = end_col - start_col
    dif_x = end_row - start_row

    for row in range(start_row,end_row):
        for col in range(start_col,end_col):
            y = int(100-dif_y/2 + col - start_col)
            x = int(100-dif_x/2 + row - start_row)
            image[x,y] = img[row,col]

    cv2.imwrite(path,image)

def compress_image():
    for dirpath,dirnames,filenames in os.walk('/home/yyz/number-plate/samples/'):
        for filename in filenames:
            img = cv2.imread(dirpath+filename)
            for y in range(175, 200):
                for x in range(175, 200):
                    print (x,y),filename
                    image = cv2.resize(img, (x,y))
                    folder_name = filename.split('.')[0]
                    if os.path.exists(dirpath+folder_name):
                        pass
                    else:
                        os.mkdir(dirpath+folder_name)
                    path = dirpath + folder_name + '/' + str(int(time.time()*1000)) + '.jpg'
                    standard_size(image,path)

def gray():
    paths = os.listdir('/home/yyz/number-plate/image/')
    for path in paths:
        img_dirs = os.listdir('/home/yyz/number-plate/image/' + path)
        for img_dir in img_dirs:
            img = cv2.imread('/home/yyz/number-plate/image/' + path + '/' + img_dir)
            grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # if os.path.exists('/home/yyz/number-plate/grey-samples/' + img_dir):
            #     pass
            # else:
            #     os.mkdir('/home/yyz/number-plate/grey-samples/' + path)
            image = cv2.resize(grey, (64, 64))
            cv2.imwrite('/home/yyz/number-plate/image/' + path + '/' + img_dir, image)

# paths = glob.glob(r'/home/yyz/number-plate/grey-samples/character/*.jpg')
# for path in paths:
#     img = cv2.imread(path)
#     image = cv2.resize(img,(150,150))
#     cv2.imwrite(path, image)
def find_point(img, x, rows):
    for row in range(rows-1,0,-1):
        if img[row,x]==0:
            y = row
            break
    return y

def calculate_angel(x1, y1, x2, y2):
    x = atan2(abs(y2-y1), abs(x2-x1))
    return x

def rotate_number(img):
    rows, cols = img.shape[:2]
    col = 0
    row = 0
    while col<cols:
        y = img[:,col]
        if np.count_nonzero(y==0)>0:
            start_col = col
            end_col = cols - 1
            while end_col>start_col:
                y = img[:,end_col]
                if np.count_nonzero(y==0)>0:
                    break
                end_col-=1
            break
        col+=1
    while row<rows:
        y = img[row,:]
        if np.count_nonzero(y==0)>0:
            start_row = row
            end_row = rows - 1
            while end_row>start_row:
                y = img[end_row,:]
                if np.count_nonzero(y==0)>0:
                    break
                end_row-=1
            break
        row+=1
    print start_col,end_col,start_row,end_row

    for row in range(rows):
        if img[row,start_col]==0:
            start_col_point = (row,start_col)
        if img[row,end_col]==0:
            end_col_point = (row,end_col)
    for col in range(cols):
        if img[start_row,col]==0:
            start_row_point = (start_row,col)
        if img[end_row,col]==0:
            end_row_point = (end_row,col)
    print start_col_point,end_col_point,start_row_point,end_row_point

    if abs(start_col_point[1]-end_row_point[1]) >= abs(end_col_point[1]-end_row_point[1]):
        x1,x2,x3,x4,x5 = [start_col_point[1] + x*abs(start_col_point[1]-end_row_point[1])/4 for x in range(5)]
        y1,y5 = start_col_point[0], end_row_point[0]
        angle_dir = 1
    elif abs(start_col_point[1]-end_row_point[1]) < abs(end_col_point[1]-end_row_point[1]):
        x1,x2,x3,x4,x5 = [end_row_point[1] + x*abs(end_col_point[1]-end_row_point[1])/4 for x in range(5)]
        y1,y5 = end_row_point[0], end_col_point[0]
        angle_dir = -1
    y2 = find_point(img, x2, rows)
    y3 = find_point(img, x3, rows)
    y4 = find_point(img, x4, rows)
    p1,p2,p3,p4,p5 = (x1, y1), (x2, y2), (x3, y3), (x4, y4), (x5, y5)
    print p1,p2,p3,p4,p5
    combs = combinations([p1, p2, p3, p4, p5], 3)
    for comb in combs:
        value = []
        for ((x1,y1),(x2,y2)) in combinations(comb, 2):
            value.append(calculate_angel(x1, y1, x2, y2))
        v1, v2, v3 = value
        if abs(v1-v2)<=0.05 and abs(v1-v3)<=0.05 and abs(v2-v3)<=0.05:
            angle = (v1 + v2 + v3)/3*angle_dir
            break

    if angle:
        print 'Angel Is %s' % angle
    else:
        print "Can't recognize the picture"
        return

    M = cv2.getRotationMatrix2D(start_col_point, angle*57.3, 1)
    dst = cv2.warpAffine(img, M, (cols, rows))
    return dst


def remove_noise():
    img = cv2.imread('./plates/1.jpg')
    rows, cols = img.shape[:2]
    for row in range(rows):
        for col in range(cols):
            b,g,r = img[row,col]
            if b==255:
                img[row,col] = np.array((0,0,0))
            else:
                img[row,col] = np.array((255,255,255))
    cv2.imwrite('./plates/1.jpg', img)

def remove_noise_(path):
    img = cv2.imread(path)
    rows, cols = img.shape[:2]
    for row in range(rows):
        for col in range(cols):
            b,g,r = img[row,col]
            if b>100 and b<=255 and g<b-40 and r<g:
                img[row,col] = np.array((0,0,0))
            else:
                img[row,col] = np.array((255,255,255))
    cv2.imwrite(path, img)


img = cv2.imread('./plates/1.JPG',0)
label_img, label_points = find_grey_connected_area(img, color_diff=20, backgroud=255)
max_num = 0
for label, points in label_points.items():
    if len(points)>max_num:
        max_num = len(points)
        max_points = points
out = np.empty_like(img)
out.fill(255)
for row,col in max_points:
    out[row,col] = 0

out = rotate_number(out)
plt.imshow(out)
plt.show()
cv2.imwrite('out.jpg', out)
