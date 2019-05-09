import cv2
import numpy as np
from skimage import data
from skimage.feature import local_binary_pattern

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

    count = np.zeros(shape=(image.shape))
    for i in range(image.shape[0]):
        count[i] = getSum(image[i])

    return np.int64(count).flatten()

if __name__ == '__main__':
    # 这里换了
    # lbp = LBP(cutImage(img, 16, 16))
    img = data.load('E:\pycharmbuild\EmotionRecognition\code\\face\jaffe\KA.AN1.39.tiff', as_gray=True)
    lbp = cutImage(local_binary_pattern(img, 8, 2), 12, 12)

    # print(local_binary_pattern(img, 8, 2))
    #
    # print(lbp)
    lbp = hist(lbp)
    print(lbp)
