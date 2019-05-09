import cv2
import numpy as np
import os
# from featureExtraction import LBP
from skimage.feature import local_binary_pattern
from model import DNN_Model, SVM_Model
from skimage import data

#text
def LBP(image):
    H, W = image.shape[0], image.shape[1]  # 获得图像长宽
    xx = [-1, 0, 1, 1, 1, 0, -1, -1]
    yy = [-1, -1, -1, 0, 1, 1, 1, 0]  # xx, yy 主要作用对应顺时针旋转时,相对中点的相对值.

    # 创建0数组,显而易见维度原始图像的长宽分别减去2，并且类型一定的是uint8,无符号8位,opencv图片的存储格式.
    res = np.zeros(shape=(H - 2, W - 2), dtype="uint8")
    for row in range(1, H - 1):
        for col in range(1, W - 1):
            str_lbp_tmp = ""  # 拼接二进制字符串
            for m in range(0, 8):
                Xtemp = xx[m] + col
                Ytemp = yy[m] + row  # 分别获得对应坐标点

                if image[Ytemp, Xtemp] > image[row, col]:  # 像素比较
                    str_lbp_tmp = str_lbp_tmp + '1'
                else:
                    str_lbp_tmp = str_lbp_tmp + '0'
            res[row - 1][col - 1] = int(str_lbp_tmp, 2)
    return res

def cutImage(image,m,n):
    h, y = image.shape #获取图片的规格
    # z = 0
    code_h, code_w = h // m, y // n
    codeimage = np.zeros((m * n, code_h * code_w))

    for i in range(m):
        for j in range(n):
            codeimage[i*m+j] = np.reshape((image[i*code_h: (i+1)*code_h, j*code_w: (j+1)*code_w]), (1, -1))

    # return np.int64(codeimage), code_h, code_w
    return np.int64(codeimage)

def getSum(counts):
    lincount = np.zeros(shape=(len(counts)))
    for i in range(len(counts)):
        lincount[counts[i]] += 1
    return np.cumsum(lincount)

def hist(image):

    count = np.zeros(shape=image.shape)
    for i in range(image.shape[0]):
        count[i] = getSum(image[i])

    return np.int64(count).flatten()
# def hist(image, m, n):
#     lcounts = np.zeros(m*n)
#     # lTemp = np.zeros(m*n)
#     # map = np.zeros(m*n)
#     # Result_image = np.zeros(shape=(image.shape[0], image.shape[1]))
#     # Result_image2 = np.zeros(shape=(image.shape[0], image.shape[1]))
#     # 计算各灰度值个数
#     for i in range(image.shape[0]):
#         for j in range(image.shape[1]):
#             lcounts[image[i, j]] += 1
#
#     # lcounts_Masked = np.ma.masked_equal(lcounts, 0)
#     # 累加各灰度值个数
#     lTemp = np.cumsum(lcounts)
#     #
#     #
#     # # lTemp2 = np.cumsum(lcounts_Masked)
#     # # 进行均衡化处理
#     # for i in range(len(lTemp)):
#     #     lcounts[i] = 255 * (lTemp[i] - np.min(lcounts)) / (np.size(image) - np.min(lcounts))
#     #     # lcounts_Masked[i] = 255 * (lTemp2[i] - np.min(lcounts_Masked)) / (np.size(image) - np.min(lcounts_Masked))
#     # map = lcounts
#     # # map2 = np.ma.filled(lcounts_Masked, 0)
#     # # 均衡化的值对应替换到原图中
#     # for i in range(image.shape[0]):
#     #     for j in range(image.shape[1]):
#     #         Result_image[i, j] = map[image[i, j]]
#     #         # Result_image2[i, j] = map2[image[i, j]]
#     return np.int64(lTemp)

if __name__ == '__main__':

    dir = "jaffe"
    filelist = os.listdir(dir)
    face = []
    emotion = []
    feature = []
    for file in filelist:
        img = data.load('E:/pycharmbuild/EmotionRecognition/code/face/' + dir + '/' + file, as_gray=True)

        # 这里换了
        # lbp = LBP(cutImage(img, 16, 16))
        lbp_data = hist((cutImage(local_binary_pattern(img, 8, 2), 16, 16)))
        feature.append(lbp_data)
        face.append(file[:2])
        emotion.append(file[3:5])


    feature = np.array(feature)
    face = np.array(face)
    emotion = np.array(emotion)

    emoton_lable = {'AN': 0, 'DI': 1, 'FE': 2, 'HA': 3, 'NE': 4, 'SA': 5, 'SU': 6}      # lable = {'KA', 'KL', 'KM', 'KR', 'MK', 'NA', 'NM', 'TM', 'UY', 'YM'}
    y_id = np.zeros(230)
    y = np.zeros((230, 7))
    for i in range(230):
        y_id[i] = emoton_lable[emotion[i]]
        y[i][emoton_lable[emotion[i]] % 7] = 1

    #DNN
    shuffle_index = np.random.permutation(230)
    x, y = feature[shuffle_index], y[shuffle_index]  #x, y = feature[shuffle_index], face[shuffle_index]     # x, y = feature[shuffle_index], emotion[shuffle_index]
    x_train, y_train = x[:200], y[:200]
    x_test, y_test = x[200:], y[200:]   # x, y
    print(np.shape(x_train))

    model = DNN_Model(65536, 50, 7)  # model = DNN_Model(25600, 500, 11)    # 64516
    model.train(x_train, y_train, epochs=40000)
    print(model.evaluate(x_test, y_test))
    model.saveModel("DNN_model_19508.h5")





    # SVM

    # shuffle_index = np.random.permutation(230)
    # x, y = feature[shuffle_index], y_id[shuffle_index]
    # x_train, y_train_id = x[:200], y[:200]
    # x_test, y_test_id = x[200:], y[200:]
    #
    # model = SVM_Model()
    # model.train(x_train, y_train_id)
    # print(model.evaluate(model.predict(x_test), y_test_id))









