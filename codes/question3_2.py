from PIL import Image , ImageOps
import matplotlib.pyplot as plt
import numpy as np
import cv2
#second -abraham

image_path = '/Users/digitcrom/Desktop/multimedia/Multimedia_HW2/abraham.jpg'  
image = Image.open(image_path)
plt.imshow(image)
plt.axis('off')
plt.title('Original Image')
plt.show()

# Split the channels using PIL
r, g, b = image.split()
r_hist = np.array(r).flatten()
g_hist = np.array(g).flatten()
b_hist = np.array(b).flatten()
image_hist = np.array(image).flatten()
fig, axes = plt.subplots(2, 2, figsize=(16, 10))

axes[0, 0].hist(r_hist, bins=256, color='red', alpha=0.6)
axes[0, 0].set_title('Red Channel Histogram')

axes[0, 1].hist(g_hist, bins=256, color='green', alpha=0.6)
axes[0, 1].set_title('Green Channel Histogram')

axes[1, 0].hist(b_hist, bins=256, color='blue', alpha=0.6)
axes[1, 0].set_title('Blue Channel Histogram')

axes[1, 1].hist(image_hist, bins=256, color='gray', alpha=0.6)
axes[1, 1].set_title('Combined Histogram')

plt.show()

gray_image =image.convert('L')
equalized_image = ImageOps.equalize(gray_image)

# Display the original and equalized images
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.imshow(gray_image, cmap='gray')
plt.title('Grayscale Image')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(equalized_image, cmap='gray')
plt.title('Histogram Equalized Image')
plt.axis('off')
plt.show()

hist_original = gray_image.histogram()
hist_equalized = equalized_image.histogram()

# Plotting the histograms
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(hist_original, color='gray')
plt.title('Histogram of Original grayscale Image')
plt.xlabel('Pixel Intensity')
plt.ylabel('Frequency')
plt.subplot(1, 2, 2)
plt.plot(hist_equalized, color='gray')
plt.title('Histogram of Equalized Image')
plt.xlabel('Pixel Intensity')
plt.ylabel('Frequency')
plt.tight_layout()
plt.show()

