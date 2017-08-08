import os
import cv2
import time
import numpy as np
from recognize import recognize

def remove_noise():
    img = cv2.imread('img.jpg')
    rows, cols = img.shape[:2]
    for row in range(rows):
        for col in range(cols):
            r,g,b = img[row,col]
            if b >= 160:
                r=b=g=0
            else:
                r=g=b=255
            img[row,col] = np.array((b,g,r))
    return img

def split_pic():
    img = remove_noise()
    rows, cols = img.shape[:2]
    split_img = []
    col = cols - 1
    while col > 0:
        end_col = col
        y = img[:,col][:,2]
        if np.count_nonzero(y!=255) > 5:
            start_col = col
            end_col = col -1
            while end_col > 0:
                y = img[:,end_col][:,2]
                if np.count_nonzero(y!=255)==0:
                    break
                end_col-=1
            if np.count_nonzero(img[:,end_col:start_col][:,:,2]!=255) > 200:
                split_img.append(img[:,end_col:start_col])
        col = end_col - 1
        if len(split_img)==6:
            split_img.append(img[:,0:end_col])
            break
    print 'get %s pieces' % len(split_img)
    return split_img

def standard_size(img):
    image = np.empty([200,200,3], dtype='float32')
    image.fill(255)

    rows, cols = img.shape[0:2]
    col=0
    row=0
    while col<cols:
        y = img[:,col][:,2]
        if np.count_nonzero(y!=255)>5:
            start_col = col
            end_col = cols - 1
            while end_col>start_col:
                y = img[:,end_col][:,2]
                if np.count_nonzero(y!=255)>5:
                    break
                end_col-=1
            break
        col+=1
    while row<rows:
        x = img[row,:][:,2]
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
            horizontal = int(100-dif_y/2 + col - start_col)
            vertical = int(100-dif_x/2 + row - start_row)
            image[vertical,horizontal] = img[row,col]

    return cv2.resize(image,(64,64))

imgs = []
for img in split_pic():
    image = standard_size(img)
    grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    imgs.append(grey)
for img in imgs[:-1]:
    from matplotlib import pyplot as plt
    plt.imshow(img)
    plt.show()
    result = recognize(np.asarray(img.reshape(1,64,64,1), dtype='float32'), 'weights_mix.h5', 65, 1)
    print result
plt.imshow(imgs[-1])
plt.show()
result = recognize(np.asarray(imgs[-1].reshape(1,64,64,1), dtype='float32'), 'weights.h5', 31, 2)
print result
