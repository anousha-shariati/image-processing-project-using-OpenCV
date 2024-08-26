from PIL import Image, ImageFilter
import matplotlib.pyplot as plt
import cv2 
import numpy as np

# Load the image
image_path = '/Users/digitcrom/Desktop/multimedia/Multimedia_HW2/AerialView.jpeg' 
image = Image.open(image_path)

# Apply Gaussian Filter
gaussian_filtered = image.filter(ImageFilter.GaussianBlur(radius=2))

# Apply Gaussian with different radius
blurred_radius_2 = image.filter(ImageFilter.GaussianBlur(radius=2))
blurred_radius_5 = image.filter(ImageFilter.GaussianBlur(radius=5))
blurred_radius_10 = image.filter(ImageFilter.GaussianBlur(radius=10))

# Display the original and blurred images
plt.figure(figsize=(20, 5))

plt.subplot(2, 2, 1)
plt.imshow(image)
plt.title('Original Image')
plt.axis('off')

plt.subplot(2, 2, 2)
plt.imshow(blurred_radius_2)
plt.title('Gaussian Blur Radius 2')
plt.axis('off')

plt.subplot(2, 2, 3)
plt.imshow(blurred_radius_5)
plt.title('Gaussian Blur Radius 5')
plt.axis('off')

plt.subplot(2, 2, 4)
plt.imshow(blurred_radius_10)
plt.title('Gaussian Blur Radius 10')
plt.axis('off')

plt.tight_layout()
plt.show()


# Apply Median Filter
median_filtered = image.filter(ImageFilter.MedianFilter(size=3))

# Apply median filter with different sizes
median_filtered_size_3 = image.filter(ImageFilter.MedianFilter(size=3))
median_filtered_size_5 = image.filter(ImageFilter.MedianFilter(size=5))
median_filtered_size_9 = image.filter(ImageFilter.MedianFilter(size=9))

# Display the original and median filtered images
plt.figure(figsize=(20, 5))

plt.subplot(2, 2, 1)
plt.imshow(image)
plt.title('Original Image')
plt.axis('off')

plt.subplot(2, 2, 2)
plt.imshow(median_filtered_size_3)
plt.title('Median Filter Size 3')
plt.axis('off')

plt.subplot(2, 2, 3)
plt.imshow(median_filtered_size_5)
plt.title('Median Filter Size 5')
plt.axis('off')

plt.subplot(2, 2, 4)
plt.imshow(median_filtered_size_9)
plt.title('Median Filter Size 9')
plt.axis('off')

plt.tight_layout()
plt.show()

# Apply Sharpening Filter
sharpened = image.filter(ImageFilter.SHARPEN)

# Define different kernels
sharpen_kernel = ImageFilter.Kernel(
    size=(3, 3),
    kernel=[
        -1, -1, -1,
        -1,  9, -1,
        -1, -1, -1
    ],
    scale=None,
    offset=0
)

edge_kernel = ImageFilter.Kernel(
    size=(3, 3),
    kernel=[
        0, -1,  0,
       -1,  4, -1,
        0, -1,  0
    ],
    scale=None,
    offset=0
)

blur_kernel = ImageFilter.Kernel(
    size=(3, 3),
    kernel=[
        1/9, 1/9, 1/9,
        1/9, 1/9, 1/9,
        1/9, 1/9, 1/9
    ],
    scale=None,
    offset=0
)

# Apply the filters
sharpened_image = image.filter(sharpen_kernel)
edge_image = image.filter(edge_kernel)
blurred_image = image.filter(blur_kernel)

# Display the images
plt.figure(figsize=(20, 5))

plt.subplot(2, 2, 1)
plt.imshow(image)
plt.title('Original Image')
plt.axis('off')

plt.subplot(2, 2, 2)
plt.imshow(sharpened_image)
plt.title('Sharpened Image')
plt.axis('off')

plt.subplot(2, 2, 3)
plt.imshow(edge_image)
plt.title('Edge Detection')
plt.axis('off')

plt.subplot(2, 2, 4)
plt.imshow(blurred_image)
plt.title('Blurred Image')
plt.axis('off')

plt.tight_layout()
plt.show()

# Plot filtered images
fig, axes = plt.subplots(1, 4, figsize=(20, 5))

# Original image
axes[0].imshow(image)
axes[0].set_title('Original Image')
axes[0].axis('off')

# Gaussian Filtered image
axes[1].imshow(gaussian_filtered)
axes[1].set_title('Gaussian Filtered')
axes[1].axis('off')

# Median Filtered image
axes[2].imshow(median_filtered)
axes[2].set_title('Median Filtered')
axes[2].axis('off')

# Sharpened image
axes[3].imshow(sharpened)
axes[3].set_title('Sharpened')
axes[3].axis('off')

plt.show()

# Convert images to numpy arrays for edge detection
original_np = np.array(image)
gaussian_np = np.array(gaussian_filtered)
median_np = np.array(median_filtered)
sharpened_np = np.array(sharpened)

# Convert to grayscale for edge detection
original_gray = cv2.cvtColor(original_np, cv2.COLOR_BGR2GRAY)
gaussian_gray = cv2.cvtColor(gaussian_np, cv2.COLOR_BGR2GRAY)
median_gray = cv2.cvtColor(median_np, cv2.COLOR_BGR2GRAY)
sharpened_gray = cv2.cvtColor(sharpened_np, cv2.COLOR_BGR2GRAY)

# Apply Sobel edge detection
sobel_original = cv2.Sobel(original_gray, cv2.CV_64F, 1, 0, ksize=3) + cv2.Sobel(original_gray, cv2.CV_64F, 0, 1, ksize=3)
sobel_gaussian = cv2.Sobel(gaussian_gray, cv2.CV_64F, 1, 0, ksize=3) + cv2.Sobel(gaussian_gray, cv2.CV_64F, 0, 1, ksize=3)
sobel_median = cv2.Sobel(median_gray, cv2.CV_64F, 1, 0, ksize=3) + cv2.Sobel(median_gray, cv2.CV_64F, 0, 1, ksize=3)
sobel_sharpened = cv2.Sobel(sharpened_gray, cv2.CV_64F, 1, 0, ksize=3) + cv2.Sobel(sharpened_gray, cv2.CV_64F, 0, 1, ksize=3)


# Function to apply Sobel and convert results for display
def apply_sobel_and_convert(image, ksize):
    sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=ksize)
    sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=ksize)
    sobel = sobel_x + sobel_y
    sobel = np.absolute(sobel)
    sobel = np.uint8(sobel / np.max(sobel) * 255)
    return sobel

# Apply Sobel with kernel size 5
sobel_original_k5 = apply_sobel_and_convert(original_gray, 5)
sobel_gaussian_k5 = apply_sobel_and_convert(gaussian_gray, 5)
sobel_median_k5 = apply_sobel_and_convert(median_gray, 5)
sobel_sharpened_k5 = apply_sobel_and_convert(sharpened_gray, 5)

# Apply Sobel with kernel size 7
sobel_original_k7 = apply_sobel_and_convert(original_gray, 7)
sobel_gaussian_k7 = apply_sobel_and_convert(gaussian_gray, 7)
sobel_median_k7 = apply_sobel_and_convert(median_gray, 7)
sobel_sharpened_k7 = apply_sobel_and_convert(sharpened_gray, 7)

# Plotting all images on one page
plt.figure(figsize=(18, 12))

plt.subplot(2, 4, 1)
plt.imshow(sobel_original_k5, cmap='gray')
plt.title('Original - K5')
plt.axis('off')

plt.subplot(2, 4, 2)
plt.imshow(sobel_gaussian_k5, cmap='gray')
plt.title('Gaussian - K5')
plt.axis('off')

plt.subplot(2, 4, 3)
plt.imshow(sobel_median_k5, cmap='gray')
plt.title('Median - K5')
plt.axis('off')

plt.subplot(2, 4, 4)
plt.imshow(sobel_sharpened_k5, cmap='gray')
plt.title('Sharpened - K5')
plt.axis('off')

plt.subplot(2, 4, 5)
plt.imshow(sobel_original_k7, cmap='gray')
plt.title('Original - K7')
plt.axis('off')

plt.subplot(2, 4, 6)
plt.imshow(sobel_gaussian_k7, cmap='gray')
plt.title('Gaussian - K7')
plt.axis('off')

plt.subplot(2, 4, 7)
plt.imshow(sobel_median_k7, cmap='gray')
plt.title('Median - K7')
plt.axis('off')

plt.subplot(2, 4, 8)
plt.imshow(sobel_sharpened_k7, cmap='gray')
plt.title('Sharpened - K7')
plt.axis('off')

plt.tight_layout()
plt.show()


# Apply Canny edge detection
canny_original = cv2.Canny(original_gray, 100, 200)
#canny_gaussian = cv2.Canny(gaussian_gray, 100, 200)
canny_gaussian = cv2.Canny(gaussian_gray, 40, 100)
canny_median = cv2.Canny(median_gray, 100, 150)
canny_sharpened = cv2.Canny(sharpened_gray, 180, 200)

canny_original_low = cv2.Canny(original_gray, 50, 100)
canny_original_high = cv2.Canny(original_gray, 100, 150)

canny_gaussian_low = cv2.Canny(gaussian_gray, 50, 100)  # Lower thresholds
canny_gaussian_high = cv2.Canny(gaussian_gray, 100, 200)  # Higher thresholds


canny_median_low = cv2.Canny(median_gray, 50, 100)  # Lower thresholds
canny_median_high = cv2.Canny(median_gray, 100, 200)  # Higher thresholds


canny_sharpened_low = cv2.Canny(sharpened_gray, 50, 150)  # Moderate thresholds
canny_sharpened_high = cv2.Canny(sharpened_gray, 100, 200)  # Higher thresholds

# Create a figure and a set of subplots
fig, axs = plt.subplots(4, 2, figsize=(18, 12))

# Set titles and turn off axes for each subplot
axs[0, 0].imshow(canny_gaussian_low, cmap='gray')
axs[0, 0].set_title('Canny - Gaussian Low Thresholds')
axs[0, 0].axis('off')

axs[0, 1].imshow(canny_gaussian_high, cmap='gray')
axs[0, 1].set_title('Canny - Gaussian High Thresholds')
axs[0, 1].axis('off')

axs[1, 0].imshow(canny_median_low, cmap='gray')
axs[1, 0].set_title('Canny - Median Low Thresholds')
axs[1, 0].axis('off')

axs[1, 1].imshow(canny_median_high, cmap='gray')
axs[1, 1].set_title('Canny - Median High Thresholds')
axs[1, 1].axis('off')

axs[2, 0].imshow(canny_sharpened_low, cmap='gray')
axs[2, 0].set_title('Canny - Sharpened Moderate Thresholds')
axs[2, 0].axis('off')

axs[2, 1].imshow(canny_sharpened_high, cmap='gray')
axs[2, 1].set_title('Canny - Sharpened High Thresholds')
axs[2, 1].axis('off')

axs[3, 0].imshow(canny_original_low, cmap='gray')
axs[3, 0].set_title('Canny - original low')
axs[3, 0].axis('off')

axs[3, 1].imshow(canny_original_high, cmap='gray')
axs[3, 1].set_title('Canny - original high')
axs[3, 1].axis('off')


plt.tight_layout()  # Adjust layout to prevent overlap
plt.show()  # Display the plot

# Plot Sobel edge detection results
fig, axes = plt.subplots(1, 4, figsize=(20, 5))

axes[0].imshow(sobel_original, cmap='gray')
axes[0].set_title('Sobel - Original')
axes[0].axis('off')

axes[1].imshow(sobel_gaussian, cmap='gray')
axes[1].set_title('Sobel - Gaussian')
axes[1].axis('off')

axes[2].imshow(sobel_median, cmap='gray')
axes[2].set_title('Sobel - Median')
axes[2].axis('off')

axes[3].imshow(sobel_sharpened, cmap='gray')
axes[3].set_title('Sobel - Sharpened')
axes[3].axis('off')

plt.show()

# Plot Canny edge detection results
fig, axes = plt.subplots(1, 4, figsize=(20, 5))

axes[0].imshow(canny_original, cmap='gray')
axes[0].set_title('Canny - Original')
axes[0].axis('off')

axes[1].imshow(canny_gaussian, cmap='gray')
axes[1].set_title('Canny - Gaussian')
axes[1].axis('off')

axes[2].imshow(canny_median, cmap='gray')
axes[2].set_title('Canny - Median')
axes[2].axis('off')

axes[3].imshow(canny_sharpened, cmap='gray')
axes[3].set_title('Canny - Sharpened')
axes[3].axis('off')

plt.show()
