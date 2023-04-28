import numpy as np
import pytesseract
import re

def distance(x1, y1, x2, y2):
    return np.sqrt((x1-x2)**2 + (y1-y2)**2)

class PagesAnalysis(): 
    """
    Get images list then finding which page is balance, income_statement, cash_flow_statement.

    Args:
        images(list): images list is converted from pdf
        ocr: ocr tool to detect text
        tesseract_config: use to config tesseract
    
    Returns:
        balance(list): include pages about balance
        income_statement(list): include pages about income statement
        cash_flow_statement(list): include pages about cash flow statement
    """
    def __init__(self, images, ocr, tesseract_config):
        self.images = images
        self.ocr = ocr
        self.tesseract_config = tesseract_config
        self.pattern = r'.{0,7}B\s?(0[1-9])a?[-/][A-Z-/]*'

    def __call__(self):
        #scans image by image, look at the top-right text, if text match regex pattern then takes pages and group in pattern
        pages = []
        for image, idx in zip(self.images, range(len(self.images))):
            image = np.array(image)
            height = 0
            width = image.shape[1]
            output = self.ocr.ocr(image, det=True, rec=False, cls=False)[0]
            cur_dis = 1e9
            res_x1, res_y1, res_x2, res_y2 = 0, 0, 0, 0
            for box in output:
                x1, y1, x2, y2= int(box[0][0]), int(box[0][1]), int(box[2][0]), int(box[2][1])
                dis = distance(x2, y1, width, height)
                if dis < cur_dis:
                    res_x1, res_y1, res_x2, res_y2 = x1, y1, x2, y2
                    cur_dis = dis
            img = image[res_y1-10:res_y2+10, res_x1-10:res_x2+10]
            text = pytesseract.image_to_string(img,lang='vie', config=self.tesseract_config)[:-2]
            result = re.search(self.pattern, text)
            if result:
                pages.append([int(result.group(1)), idx])

        # balance, income_statement, cash_flow_statement has result.group() increase 1 unit, respectively
        code = 0
        code_table = 0
        balance = []
        income_statement = []
        cash_flow_statement = []
        for i in range(len(pages)):
            if pages[i][0] != code:
                code_table+=1
                code = pages[i][0]
            if code_table == 4:
                break
            if code_table == 1:
                balance.append(pages[i][1])
            if code_table == 2:
                income_statement.append(pages[i][1])
            if code_table == 3:
                cash_flow_statement.append(pages[i][1])

        return balance, income_statement, cash_flow_statement