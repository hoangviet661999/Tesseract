import pytesseract
import re
import imutils
import cv2

def prepocess_image(image, sig_model):
    image_height = image.shape[0]
    image_width = image.shape[1]
    osd = pytesseract.image_to_osd(image)
    angle = re.search('(?<=Rotate: )\d+', osd).group(0)
    image = imutils.rotate_bound(image, angle=int(angle))
    image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    #masked signature
    image_rs = cv2.resize(image, (640,640))
    results = sig_model(image_rs)
    boxes = results.xyxy[0]
    if len(boxes)==1 and boxes[0][4]>0.5:
        box = boxes[0] 
        x1 = int(box[0]/640*image_width)
        y1 = int(box[1]/640*image_height)
        x2 = int(box[2]/640*image_width)
        y2 = int(box[3]/640*image_height)
        image[y1:y2, x1:x2] = 255 

    image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

    return image