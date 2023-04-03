import cv2
import pytesseract
from pdf2image import convert_from_path

input_path = "example/mbb_2022.pdf"
images = convert_from_path(input_path)

# for i in range(len(images)):
#     images[i].save('page'+ str(i) +'.jpg', 'JPEG')

i=3
images[i].save('page'+ str(i) +'.jpg', 'JPEG')

img = cv2.imread('page'+ str(i) +'.jpg')

custom_config = r'--oem 3 --psm 6'
print(pytesseract.image_to_string(img, config=custom_config))