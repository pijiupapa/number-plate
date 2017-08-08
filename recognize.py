# -*- coding:utf-8 -*-

import os
import keras
import numpy as np

from PIL import Image
from keras import backend as K
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D, Convolution2D, Activation

map_dict = {'0':0,'1':1,'2':2,'3':3,'4':4,'5':5,'6':6,'7':7,'8':8,'9':9,'a':10,
            'b':11,'c':12,'d':13,'e':14,'f':15,'g':16,'h':17,'j':18,'k':19,
            'l':20,'m':21,'n':22,'p':23,'q':24,'r':25,'s':26,'t':27,'u':28,
            'v':29,'w':30,'x':31,'y':32,'z':33,'冀':34,'豫':35,'云':36,'辽':37,
            '黑':38,'湘':39,'皖':40,'鲁':41,'新':42,'苏':43,'浙':44,'赣':45,'鄂':46,
            '桂':47,'甘':48,'晋':49,'蒙':50,'陕':51,'吉':52,'闽':53,'贵':54,'粤':55,
            '川':56,'青':57,'藏':58,'琼':59,'宁':60,'渝':61,'京':62,'津':63,'沪':64}
inversed_map_one = {v:k for k,v in map_dict.items()}

map_ch = {'冀':1,'豫':2,'云':3,'辽':4,'黑':5,'湘':6,'皖':7,'鲁':8,'新':9,
        '苏':10,'浙':11,'赣':12,'鄂':13,'桂':14,'甘':15,'晋':16,'蒙':17,'陕':18,
        '吉':19,'闽':20,'贵':21,'粤':22,'川':23,'青':24,'藏':25,'琼':26,'宁':27,
        '渝':28,'京':29,'津':30,'沪':0}
inversed_map_two = {v:k for k,v in map_ch.items()}

def load_data(path):
    img_dirs = os.listdir(path)
    data_list = []
    label_list = []
    for img_dir in img_dirs:
        imgs = os.listdir(path + img_dir)
        num = len(imgs)
        data_tmp = np.empty((num,1,64,64), dtype='float32')
        label_tmp = np.empty((num,),dtype='float32')
        for i in range(num):
            print img_dir,imgs[i]
            img = Image.open(path + '/' + img_dir + '/' + imgs[i])
            arr = np.asarray(img, dtype='float32')
            data_tmp[i,:,:,:] = arr
            label_tmp[i] = map_ch[img_dir]
        data_list.append(data_tmp)
        label_list.append(label_tmp)
    data_total = np.vstack(data_list)
    label_total = np.hstack(label_list)
    print data_total.shape,label_total.shape
    return data_total, label_total

def load_test(path):
    img_dirs = os.listdir(path)
    data_list = []
    label_list = []
    for img_dir in img_dirs:
        data_tmp = np.empty((1,1,64,64), dtype='float32')
        label_tmp = np.empty((1,),dtype='float32')
        img = Image.open(path + img_dir)
        arr = np.asarray(img, dtype='float32')
        data_tmp[0,:,:,:] = arr
        label_tmp[0] = map_ch[img_dir.split('.')[0]]
        data_list.append(data_tmp)
        label_list.append(label_tmp)
    data_total = np.vstack(data_list)
    label_total = np.hstack(label_list)
    print data_total.shape,label_total.shape
    return data_total, label_total

def train():
    data ,label = load_data('/home/yyz/number-plate/grey-samples/data/')
    print data.shape[0]
    data_test, label_test = load_data('/home/yyz/number-plate/grey-samples/test/')
    print data_test.shape[0]
    np.savez('dataset_c', data=data, label=label, data_test=data_test, label_test=label_test)

    batch_size = 200
    num_classes = 31
    epochs = 10
    input_shape = (64,64,1)

    data = data.reshape(data.shape[0],64,64,1)
    label = keras.utils.to_categorical(label, num_classes)
    data_test = data_test.reshape(data_test.shape[0],64,64,1)
    label_test = keras.utils.to_categorical(label_test, num_classes)
    x_train = data
    y_train = label
    x_test = data_test
    y_test = label_test
    print x_train.shape,y_train.shape,x_test.shape,y_test.shape
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    # model.add(Dropout(0.25))

    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    # model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adadelta(),
                  metrics=['accuracy'])

    model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              verbose=1,
              validation_data=(x_test,y_test))

    model.save_weights('/home/yyz/number-plate/weights.h5')
    # model.load_weights('/home/yyz/shixin_/weights.h5')
    score = model.evaluate(x_test, y_test, verbose=0)
    print(score)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])
    result = model.predict(x_test[:4])
    for i in result:
        print np.argmax(i)
    from matplotlib import pyplot as plt
    plt.imshow(x_test[0].reshape(64,64))
    plt.show()

def recognize(img_arr, weight='weights_mix.h5', num_classes=65, map_type=1):
    batch_size = 200
    input_shape = (64,64,1)
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    # model.add(Dropout(0.25))

    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    # model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adadelta(),
                  metrics=['accuracy'])
    model.load_weights(weight)
    result = model.predict(img_arr)
    if map_type==1:
        return inversed_map_one[np.argmax(result)]
    else:
        return inversed_map_two[np.argmax(result)]
