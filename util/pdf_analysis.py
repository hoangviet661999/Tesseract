import numpy as np
import pytesseract
import re
import cv2
import imutils

class PdfAnalysis(): 
    """
    Get images list then finding which page is balance, income_statement, cash_flow_statement.

    Args:
        images(list): images list is converted from pdf
        sig_model: model to detect signature
        tab_model: model to detect table
    
    Returns:
        pages(list): list of signatures
    """
    def __init__(self, images, tab_model, sig_model):
        self.images = images
        self.sig_model = sig_model
        self.tab_model = tab_model

    def __call__(self):
        pages = []
        for i in range(len(self.images)):
            #read images and preprocessing: correcting text orientation, convert to gray image and binary it
            image = np.array(self.images[i])
            osd = pytesseract.image_to_osd(image)
            angle = re.search('(?<=Rotate: )\d+', osd).group(0)
            image = imutils.rotate_bound(image, angle=int(angle))
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            #feed images to table detection model
            results = self.tab_model(image)
            boxes = results.xyxy[0]
            if len(boxes) == 1:
                box = boxes[0] 
                x1 = int(box[0])
                y1 = int(box[1])
                x2 = int(box[2])
                y2 = int(box[3])
                area = (x2-x1)*(y2-y1)
                if area/0.4 > image.shape[0]*image.shape[1]:
                    pages.append(i)
                    break
        for i in range(pages[0], 20):
            image = np.array(self.images[i])
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            image = cv2.resize(image, (640,640))
            results = self.sig_model(image)
            boxes = results.xyxy[0]
            if len(boxes)==1 and boxes[0][4]>0.7:
                pages.append(i)
        return pages
            