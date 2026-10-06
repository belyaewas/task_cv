import numpy as np
image = np.array([
    [12,  2,  6,  8,  4],
    [ 8,  0,  2,  5, 17],
    [22,  9,  4, 21,  7],
    [15,  3,  2, 11,  8],
    [ 7, 14, 19, 20, 10]
])

# Ядро свёртки 3×3
kernel = np.array([
    [ 1, -1,  3],
    [ 2, -2,  4],
    [ 3, -3,  5]
])

kernel_blur = np.array([
    [ 1, 1,  1],
    [ 1, 1,  1],
    [ 1, 1,  1]
])
kernel_blur= (1/9) * kernel_blur
from scipy.signal import convolve2d

# Выполнение свёртки
result = convolve2d(image, kernel, mode='same', boundary='fill', fillvalue=0)
result2 = convolve2d(result, kernel_blur, mode='same', boundary='fill', fillvalue=0)
result3= result2.astype(int)+result
# Вывод результата
print(result3)