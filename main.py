:import SimpleITK as sitk
import numpy as np
import matplotlib.pyplot as plt
import cv2

radii = 512


def homomorphic_filter(src, d0=9, r1=0.5, rh=2, c=4, h=2.0, l=0.5):
    gray = src.copy()
    if len(src.shape) > 2:
        gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    gray = np.float64(gray)
    rows, cols = gray.shape

    gray_fft = np.fft.fft2(gray)
    gray_fftshift = np.fft.fftshift(gray_fft)
    dst_fftshift = np.zeros_like(gray_fftshift)
    M, N = np.meshgrid(np.arange(-cols // 2, cols // 2), np.arange(-rows // 2, rows // 2))
    D = np.sqrt(M ** 2 + N ** 2)
    Z = (rh - r1) * (1 - np.exp(-c * (D ** 2 / d0 ** 2))) + r1
    dst_fftshift = Z * gray_fftshift
    dst_fftshift = (h - l) * dst_fftshift + l
    dst_ifftshift = np.fft.ifftshift(dst_fftshift)
    dst_ifft = np.fft.ifft2(dst_ifftshift)
    dst = np.real(dst_ifft)
    dst = np.uint8(np.clip(dst, 0, 255))
    return dst

def is_in_range(i,j, radio):
    radio2 = (i-radio) * (i-radio) + (j -radio ) * (j -radio)
    if radio2 > radii*radii:
        return 0
    return 1

def histogram(image):
    (row, col) = image.shape
    hist = [0]*256
    for i in range(row):
        for j in range(col):
            hist[image[i,j]] +=1
    return hist

def exhistogram(image):
    (row, col) = image.shape
    hist = [0]*256
    for i in range(row):
        for j in range(col):
            if is_in_range(i,j, radii):
                hist[image[i,j]] +=1
    return hist

hist_balance =  [0] * 256

def balance(image):
    (row, col) = image.shape
    hist = [0] * 256
    count = int(0)
    hist_cumulate = [0] * 256

    for i in range(row):
        for j in range(col):
            if is_in_range(i, j, radii):
                hist[image[i, j]] += 1
    for k in range(0, len(hist)):
        count = count + hist[k]
        if k > 1 :
            hist_cumulate[k] =  hist_cumulate[k-1] + hist[k]
            hist_balance[k] = int(hist_cumulate[k]*254/count)

    print("count: %d" % (count))

    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            if not is_in_range(i, j, radii):
                temp = image[i, j]
                image[i, j] = hist_balance[temp]
    return image


def reduce(img):
    maxV=int(img.max())
    minV=int(img.min())
    avarage = int((maxV + minV) /2)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if img[i, j] <23:
                img[i, j] = 255
    maxV=int(img.max())
    minV=int(img.min())
    avarage = int((maxV + minV) /2)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if img[i, j] < 1:
                img[i, j] = ((img[i, j] - minV) * 255) / (maxV - minV)
    return img

def pre_handle(img):
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if not is_in_range(i,j,radii):
                img[i, j] = 0
    return img


#全局灰度线性变换
def global_linear_transmation(img): #将灰度范围设为0~255

    maxV=img[radii, radii]
    minV=img[radii, radii]

    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if is_in_range(i,j,radii):
                if img[i, j] > maxV:
                    maxV = img[i, j]
                if img[i, j] < minV:
                    minV = img[i, j]
    print("min %d" % (minV))
    print("max %d" % (maxV))
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if is_in_range(i, j, radii):
                if img[i, j] <= maxV:
                    if img[i, j] >= minV :
                        img[i,j] = ((img[i,j]-minV)*254)/(maxV-minV)
                        # img[i, j] = img[i,j]
                else:
                    print("error")
    return img

def get_window_size(window_type):
    if window_type =='lung':
        center = -500
        width =2000
    elif window_type =='abdomen':
        center =40
        width =400
    elif window_type =='bone':
        center =1229
        width =2924
    return center, width

def setDicomWinWidthWinCenter(img_data, window_type):
    img_temp = img_data
    rows =len(img_temp)
    cols =len(img_temp[0])
    print(rows)
    print(cols)
    center, width = get_window_size(window_type)
    print(center)
    print(width)

    img_temp.flags.writeable = True
    min = (2 * center - width) /2.0 +0.5
    max = (2 * center + width) /2.0 +0.5
    dFactor =255.0 / (max - min)
    print(dFactor)
    for i in np.arange(rows):
        for j in np.arange(cols):
            if img_temp[i, j] < min :
                img_temp[i,j] = 0
            elif img_temp[i, j] > max :
                img_temp[i,j] = 255
            else :
                img_temp[i, j] = int((img_temp[i, j] - min) * dFactor)
            # print("i=%d,j=%d" % (i, j))
    print("setDicomWinWidthWinCenter - Done")
    return img_temp

image = sitk.ReadImage("d://_code/dicom/00001.dcm")

image_array = np.squeeze(sitk.GetArrayFromImage(image))

# img_scaled = cv2.normalize(image_array,dst=None,alpha=0,beta=255,norm_type=cv2.NORM_MINMAX)

img_scaled = setDicomWinWidthWinCenter(image_array, 'bone')

# img_scaled1 = cv2.normalize(img_scaled,dst=None,alpha=0,beta=255,norm_type=cv2.NORM_MINMAX)

image0 = np.uint8(img_scaled)

# plt.imshow(image0, vmin=0, vmax=255,cmap = plt.cm.gray)


plt.subplot(4,2,1)
plt.imshow(image0,vmin=0, vmax=255,cmap = plt.cm.gray)
plt.title('origin image')
image_hist0 = exhistogram(image0)
plt.subplot(4,2,2)
plt.plot(image_hist0)

image1=pre_handle(image0)  #边缘处理
plt.subplot(4,2,3)
plt.imshow(image1,vmin=0, vmax=255,cmap = plt.cm.gray)
plt.imsave("d://_code/image1.jpg",image1, cmap='gray')
image_hist1 = exhistogram(image1)#统计变换后图像的各灰度值像素的个数
plt.subplot(4,2,4)
plt.plot(image_hist1)

# image1=reduce(image0)
# plt.subplot(3,2,3)
# plt.imshow(image1,vmin=0, vmax=255,cmap = plt.cm.gray)
# plt.imsave("d://_code/image1.jpg",image1, cmap='gray')
# image_hist1 = exhistogram(image1)#统计变换后图像的各灰度值像素的个数
# plt.subplot(3,2,4)
# plt.plot(image_hist1)


image2=global_linear_transmation(image1)
plt.subplot(4,2,5)
plt.imshow(image2,vmin=0, vmax=255,cmap = plt.cm.gray)
plt.imsave("d://_code/image2.jpg",image2, cmap='gray')
image_hist2 = exhistogram(image2)#统计变换后图像的各灰度值像素的个数
plt.subplot(4,2,6)
plt.plot(image_hist2)



image3 = cv2.equalizeHist(image2)
# plt.imshow(image3,vmin=0, vmax=255,cmap = plt.cm.gray)
plt.imsave("d://_code/image3.jpg",image3, cmap='gray')

# balance(image1)

image4=homomorphic_filter(image2)
plt.subplot(4,2,7)
plt.imshow(image4,vmin=0, vmax=255,cmap = plt.cm.gray)
plt.imsave("d://_code/image4.jpg",image4, cmap='gray')
image_hist4 = exhistogram(image4)#统计变换后图像的各灰度值像素的个数
plt.subplot(4,2,8)
plt.plot(image_hist4)

plt.show()

# plt.imshow(0)