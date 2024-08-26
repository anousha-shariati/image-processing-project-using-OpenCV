from PIL import Image , ImageOps
import matplotlib.pyplot as plt
import numpy as np
import cv2


image_path = '/Users/digitcrom/Desktop/multimedia/Multimedia_HW2/trees.jpeg' 
image = Image.open(image_path)

# Convert PIL Image to NumPy array for OpenCV processing
image_np = np.array(image)
# Convert RGB to BGR for OpenCV processing
image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

plt.imshow(image)
plt.axis('off')
plt.title('Original Image')
plt.show()

# Split the channels using OpenCV
B, G, R = cv2.split(image_np)

# Display the channels using matplotlib
plt.figure(figsize=(10, 3))
plt.subplot(131)
plt.imshow(R, cmap='gray')
plt.title('Red Channel')
plt.subplot(132)
plt.imshow(G, cmap='gray')
plt.title('Green Channel')
plt.subplot(133)
plt.imshow(B, cmap='gray')
plt.title('Blue Channel')
plt.show()

# Using PIL to split the image channels
r, g, b = image.split()
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
axes[0].imshow(r, cmap='Reds')
axes[0].axis('off')
axes[0].set_title('Red Channel')

axes[1].imshow(g, cmap='Greens')
axes[1].axis('off')
axes[1].set_title('Green Channel')

axes[2].imshow(b, cmap='Blues')
axes[2].axis('off')
axes[2].set_title('Blue Channel')
plt.show()

r_hist = np.array(r).flatten()
g_hist = np.array(g).flatten()
b_hist = np.array(b).flatten()
image_hist = np.array(image).flatten()
fig, axes = plt.subplots(2, 2, figsize=(18, 12))

axes[0, 0].hist(r_hist, bins=256, color='red', alpha=0.6)
axes[0, 0].set_title('Red Channel Histogram')

axes[0, 1].hist(g_hist, bins=256, color='green', alpha=0.6)
axes[0, 1].set_title('Green Channel Histogram')

axes[1, 0].hist(b_hist, bins=256, color='blue', alpha=0.6)
axes[1, 0].set_title('Blue Channel Histogram')

axes[1, 1].hist(image_hist, bins=256, color='gray', alpha=0.6)
axes[1, 1].set_title('Combined Histogram')

plt.show()

# Convert image to grayscale for a luminance histogram
gray_image = ImageOps.grayscale(image)
gray_hist = np.array(gray_image).flatten()

#Plotting
plt.hist(r_hist, bins=256, color='red', alpha=0.6, label='Red')
plt.hist(g_hist, bins=256, color='green', alpha=0.6, label='Green')
plt.hist(b_hist, bins=256, color='blue', alpha=0.6, label='Blue')
plt.hist(gray_hist, bins=256, color='gray', alpha=0.6, label='Grayscale')
plt.title('Histogram Comparison')
plt.xlabel('Pixel Intensity')
plt.ylabel('Frequency')
plt.legend()
plt.show()

