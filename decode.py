import cv2
import numpy as np
import math
from tabulate import tabulate
import os
import encode

def dequantize(matrix) :
    quantization_matrix = np.array(
        [
            [16, 11, 10, 16, 24, 40, 51, 61],
            [12, 12, 14, 19, 26, 58, 60, 55],
            [14, 13, 16, 24, 40, 57, 69, 56],
            [14, 17, 22, 29, 51, 87, 80, 62],
            [18, 22, 37, 56, 68, 109, 103, 77],
            [24, 35, 55, 64, 81, 104, 113, 92],
            [49, 64, 78, 87, 103, 121, 120, 101],
            [72, 92, 95, 98, 112, 100, 103, 99],
        ]
    )
    return (matrix * quantization_matrix).astype(int)

def idct_transform(matrix):
    idct = []
    m, n = (8, 8)
    for i in range(m):
        idct.append([None for _ in range(n)])
    for i in range(m):
        for j in range(n):
            sum = 0
            for k in range(m):
                for l in range(n):
                    if k == 0:
                        Cu = 1 / (2 ** 0.5)
                    else:
                        Cu = 1
                    if l == 0:
                        Cv = 1 / (2 ** 0.5)
                    else:
                        Cv = 1
                    idct1 = (
                        Cu
                        * Cv
                        * matrix[k][l]
                        * math.cos((2 * i + 1) * k * math.pi / 16)
                        * math.cos((2 * j + 1) * l * math.pi / 16)
                    )
                    sum += idct1
            idct[i][j] = round(1 / 4 * sum)
    return np.array(idct)

dequantized = dequantize(encode.quantization)
print("Tabel warna setelah dequantization:")
print(tabulate(dequantized, tablefmt="psql"))
print()

idct_matrix = idct_transform(dequantized)
print("Tabel warna setelah invers DCT:")
print(tabulate(idct_matrix, tablefmt="psql"))
print()

restored_image = idct_matrix + 128
restored_image = np.array(np.clip(restored_image, 0, 255), dtype=np.int16)
print("Tabel warna setelah ditambah 128:")
print(tabulate(restored_image, tablefmt="psql"))
print()

resized_image = cv2.resize(
    restored_image, (8, 8), interpolation=cv2.INTER_LINEAR
)

cv2.destroyAllWindows()
cv2.namedWindow("Restored Image", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Restored Image", 480, 480)
cv2.imshow("Restored Image", resized_image)
cv2.imwrite(f"{os.getcwd()}\\results\{encode.image_name}\decompresed.jpg", resized_image)

cv2.waitKey(0)
cv2.destroyAllWindows()