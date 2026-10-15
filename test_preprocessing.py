import cv2
import matplotlib.pyplot as plt
from tkinter import Tk
from tkinter.filedialog import askopenfilename

Tk().withdraw()

file_path = askopenfilename(
    title="Select OPG Image",
    filetypes=[
        ("Images", "*.png *.jpg *.jpeg *.bmp")
    ]
)

if not file_path:
    print("No image selected")
    exit()

img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)

coords = cv2.findNonZero(img)
x, y, w, h = cv2.boundingRect(coords)

cropped = img[y:y+h, x:x+w]

resized = cv2.resize(cropped, (512, 512))

clahe = cv2.createCLAHE(
    clipLimit=2.0,
    tileGridSize=(8,8)
)

enhanced = clahe.apply(resized)

plt.figure(figsize=(12,5))

plt.subplot(1,2,1)
plt.imshow(resized, cmap="gray")
plt.title("Before CLAHE")
plt.axis("off")

plt.subplot(1,2,2)
plt.imshow(enhanced, cmap="gray")
plt.title("After CLAHE")
plt.axis("off")

plt.show()