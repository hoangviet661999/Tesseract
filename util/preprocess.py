import pytesseract
import re
import imutils
import cv2

def prepocess_image(image):
    osd = pytesseract.image_to_osd(image)
    angle = re.search('(?<=Rotate: )\d+', osd).group(0)
    image = imutils.rotate_bound(image, angle=int(angle))
    image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

    return image