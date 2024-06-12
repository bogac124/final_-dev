import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


histogram_data = {
    100: 12, 101: 18, 102: 32, 103: 48, 104: 52, 105: 65, 106: 55, 107: 42, 108: 32, 109: 16,
    110: 10, 140: 5, 141: 18, 142: 25, 143: 32, 144: 40, 145: 65, 146: 43, 147: 32, 148: 20, 149: 10, 150: 4
}


intensity_values = np.array(list(histogram_data.keys()))
pixel_counts = np.array(list(histogram_data.values()))


total_pixels = np.sum(pixel_counts)


pdf = pixel_counts / total_pixels


cdf = np.cumsum(pdf)


within_class_variance = np.zeros_like(intensity_values, dtype=float)
for i, t in enumerate(intensity_values):
    
    p1 = cdf[i]
    p2 = 1 - p1

    
    mu1 = np.sum(intensity_values[:i + 1] * pdf[:i + 1]) / p1 if p1 > 0 else 0
    mu2 = np.sum(intensity_values[i + 1:] * pdf[i + 1:]) / p2 if p2 > 0 else 0

    
    var1 = np.sum(((intensity_values[:i + 1] - mu1) ** 2) * pdf[:i + 1]) / p1 if p1 > 0 else 0
    var2 = np.sum(((intensity_values[i + 1:] - mu2) ** 2) * pdf[i + 1:]) / p2 if p2 > 0 else 0

   
    within_class_variance[i] = p1 * var1 + p2 * var2


optimal_threshold = intensity_values[np.argmin(within_class_variance)]
print(f"Optimal Otsu Threshold: {optimal_threshold}")


file_path = 'c:\\Users\\kralb\\OneDrive\\Masaüstü\\Proglama Final ödevi\\matriks dosyasi\\soru1_2_data.xlsx'
data = pd.read_excel(file_path, header=None)


matrix_data = data.values

binary_matrix_data = (matrix_data >= optimal_threshold).astype(int)

fig, ax = plt.subplots(1, 2, figsize=(12, 6))
ax[0].imshow(matrix_data, cmap='gray')
ax[0].set_title('Original Gray Scale Image')
ax[1].imshow(binary_matrix_data, cmap='gray')
ax[1].set_title('Binary Image after Otsu Thresholding')
plt.show()


threshold_value = optimal_threshold
background_pixels_data = pixel_counts[intensity_values <= threshold_value]
foreground_pixels_data = pixel_counts[intensity_values > threshold_value]


background_weight = np.sum(background_pixels_data) / total_pixels
background_mean = np.sum(intensity_values[intensity_values <= threshold_value] * background_pixels_data) / np.sum(
    background_pixels_data)
background_variance = np.sum(
    ((intensity_values[intensity_values <= threshold_value] - background_mean) ** 2) * background_pixels_data) / np.sum(
    background_pixels_data)


foreground_weight = np.sum(foreground_pixels_data) / total_pixels
foreground_mean = np.sum(intensity_values[intensity_values > threshold_value] * foreground_pixels_data) / np.sum(
    foreground_pixels_data)
foreground_variance = np.sum(
    ((intensity_values[intensity_values > threshold_value] - foreground_mean) ** 2) * foreground_pixels_data) / np.sum(
    foreground_pixels_data)

print(f"Background Weight (Wb): {background_weight}")
print(f"Background Mean (Mb): {background_mean}")
print(f"Background Variance (Vb): {background_variance}")
print(f"Foreground Weight (Wf): {foreground_weight}")
print(f"Foreground Mean (Mf): {foreground_mean}")
print(f"Foreground Variance (Vf): {foreground_variance}")


fig, axs = plt.subplots(2, 2, figsize=(12, 12))
axs[0, 0].bar(intensity_values, pixel_counts)
axs[0, 0].set_title('Histogram')
axs[0, 1].imshow(matrix_data, cmap='gray')
axs[0, 1].set_title('Original Gray Scale Image')
axs[1, 0].bar(intensity_values[intensity_values <= threshold_value], background_pixels_data)
axs[1, 0].set_title('Background Histogram')
axs[1, 1].bar(intensity_values[intensity_values > threshold_value], foreground_pixels_data)
axs[1, 1].set_title('Foreground Histogram')
plt.show()


plt.figure(figsize=(10, 6))
plt.plot(intensity_values, within_class_variance, label='Within Class Variance')
plt.axvline(x=optimal_threshold, color='r', linestyle='--', label=f'Optimal Threshold = {optimal_threshold}')
plt.xlabel('Intensity Value')
plt.ylabel('Within Class Variance')
plt.title('Within Class Variance for Each Threshold')
plt.legend()
plt.show()

