import numpy as np
from matplotlib import pyplot as plt
from PIL import Image
from rembg import remove
import pydicom


# 统计array中数字出现的频率
def countArray(array):
    import pandas as pd
    result = pd.value_counts(array)
    return result


# 处理图片，输出mask
def getMask(path, save=None):
    input = Image.open(path)
    masked_im = remove(input, post_process_mask=True)
    masked_im = np.asarray(masked_im)

    # print(f"mask_im.shape={masked_im.shape}")
    mask = masked_im[:, :, -1]

    if None != save:
        mask_im = Image.fromarray(mask, mode='L')
        mask_im.save(save)
    return mask

# 处理arr，输出mask
def getArrayMask(arr, save=None):
    masked_im = remove(arr, post_process_mask=True)
    masked_im = np.asarray(masked_im)

    # print(f"mask_im.shape={masked_im.shape}")
    mask = masked_im[:, :, -1]

    if None != save:
        mask_im = Image.fromarray(mask, mode='L')
        mask_im.save(save)
    return mask


# 将灰度数组映射为直方图字典,nums表示灰度的数量级
def arrayToHist(grayArray, nums, use_mask=False, mask=None):
    if (len(grayArray.shape) != 2):
        print("shape error")
        return None
    h, w = grayArray.shape
    hist = {}
    sum = 0
    for k in range(nums):
        hist[k] = 0
    for i in range(h):
        for j in range(w):
            # 略过非mask区域，只统计区域内
            if use_mask and mask[i][j] == 0:
                continue
            if hist.get(grayArray[i][j]) is None:
                hist[grayArray[i][j]] = 0
            hist[grayArray[i][j]] += 1
            sum += 1

    # normalize
    if use_mask is False:
        n = w * h
    else:
        n = sum
    for key in hist.keys():
        hist[key] = float(hist[key]) / n
    return hist

# 合并dict


# 画直方图，传入的直方图要求是个字典，每个灰度对应着概率
def drawHist(hist, name):
    keys = hist.keys()
    values = hist.values()
    x_size = len(hist) - 1  # x轴长度，也就是灰度级别
    axis_params = []
    axis_params.append(0)
    axis_params.append(x_size)

    # plt.figure()
    if name != None:
        plt.title(name)
    plt.bar(tuple(keys), tuple(values))  # 绘制直方图
    # plt.show()


# 计算累计直方图计算出新的均衡化的图片，nums为灰度数,256
def equalization(grayArray, h_s, nums, use_mask=False, mask=None, replace_bg=False):
    # 计算累计直方图
    tmp = 0.0
    h_acc = h_s.copy()
    for i in range(256):
        tmp += h_s[i]
        h_acc[i] = tmp

    if (len(grayArray.shape) != 2):
        print("length error")
        return None
    h, w = grayArray.shape
    des = np.zeros((h, w), dtype=np.uint8)
    for i in range(h):
        for j in range(w):
            # 非mask区域不进行转换 or 转换为纯黑
            if use_mask:
                if mask[i][j] == 0:
                    des[i][j] = 0 if replace_bg == True else grayArray[i][j]
                else:
                    des[i][j] = int((nums - 1) * h_acc[grayArray[i][j]] + 0.5)
            else:
                des[i][j] = int((nums - 1) * h_acc[grayArray[i][j]] + 0.5)
    return des


# 直方图匹配函数，接受原始图像和目标灰度直方图
def histMatch(grayArray, h_d, use_mask=False, mask=None, replace_bg=False):
    # 从目标灰度直方图h_d中计算累计直方图h_acc
    tmp = 0.0
    h_acc = h_d.copy()
    for i in range(256):
        tmp += h_d[i]
        h_acc[i] = tmp

    # 计算原始图像h1的累计直方图h1_acc
    h1 = arrayToHist(grayArray, 256, use_mask, mask)
    tmp = 0.0
    h1_acc = h1.copy()
    for i in range(256):
        tmp += h1[i]
        h1_acc[i] = tmp

    # 计算映射
    M = np.zeros(256)
    for i in range(256):
        idx = 0
        minv = 1
        for j in h_acc:
            if (np.fabs(h_acc[j] - h1_acc[i]) < minv):
                minv = np.fabs(h_acc[j] - h1_acc[i])
                idx = int(j)
        M[i] = idx

    # 输出映射后的image
    w, h = grayArray.shape
    des = np.zeros((w, h), dtype=np.uint8)
    for i in range(w):
        for j in range(h):
            # 如果使用mask:
            # - 非mask区域不进行映射 or 转换为纯黑
            # - mask区域进行映射
            if use_mask:
                if mask[i][j] == 0:
                    des[i][j] = 0 if replace_bg == True else grayArray[i][j]
                else:
                    des[i][j] = M[grayArray[i][j]]
            # 如果不使用mask: 直接对所有像素进行转换
            else:
                des[i][j] = M[grayArray[i][j]]
    return des

def readPydicom(path):
    ds = pydicom.dcmread(path, force=True)
    data = np.array(ds.pixel_array)
    normalized = normalize(data)
    return normalized

def normalize(im_arr, nums=255):
    if len(im_arr.shape) != 2:
        print('shape error!')
        return None
    h,w = im_arr.shape
    max_pixel,min_pixel = np.max(im_arr),np.min(im_arr)
    image = (im_arr.flatten() - min_pixel) / (max_pixel - min_pixel)
    image = image.reshape(h,w)
    image = image * nums
    return image.astype(np.uint8)