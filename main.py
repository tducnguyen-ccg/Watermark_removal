import cv2
import numpy as np
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler

img = cv2.imread('images/add_watermark.png', 0)
ddepth = cv2.CV_16S
scale = 1
delta = 0

laplacian = cv2.Laplacian(img, cv2.CV_64F, ksize=7)
# sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=5)
# sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=5)
# # sobely = cv2.Sobel(img, ddepth, 0, 1, ksize=3, scale=scale, delta=delta, borderType=cv2.BORDER_DEFAULT)


hight = laplacian.shape[0]
width = laplacian.shape[1]
laplacian_filter = np.zeros([hight, width])
# plt.imshow(laplacian, cmap='gray')
# plt.show()

for h_id in range(hight):
    for w_id in range(width):
        if laplacian[h_id, w_id] >= 10000:
            # print(laplacian[h_id, w_id])
            laplacian_filter[h_id, w_id] = laplacian[h_id, w_id]

# plt.imshow(laplacian_filter, cmap='gray')
# plt.show()

# Reduce salt noise by median filter
scaler = MinMaxScaler(feature_range=(0, 255))
scaler.fit(laplacian_filter)
laplacian_filter = scaler.transform(laplacian_filter)
laplacian_filter = laplacian_filter.astype(np.uint8)

laplacian_filter_median = cv2.medianBlur(laplacian_filter, 3)


plt.subplot(2, 2, 1), plt.imshow(img, cmap='gray')
plt.title('Original'), plt.xticks([]), plt.yticks([])
plt.subplot(2, 2, 2), plt.imshow(laplacian, cmap='gray')
plt.title('Laplacian'), plt.xticks([]), plt.yticks([])
plt.subplot(2, 2, 3), plt.imshow(laplacian_filter, cmap='gray')
plt.title('Laplacian filter'), plt.xticks([]), plt.yticks([])
plt.subplot(2, 2, 4), plt.imshow(laplacian_filter_median, cmap='gray')
plt.title('Laplacian Median'), plt.xticks([]), plt.yticks([])
plt.show()

# Invert color
final_array = np.zeros([hight, width])
for h_id in range(hight):
    for w_id in range(width):
        if laplacian_filter[h_id, w_id] != 0:
            final_array[h_id, w_id] = 0
        else:
            final_array[h_id, w_id] = 255


laplacian_filter_median
plt.imshow(final_array, cmap='gray')
plt.show()

print('Done')


