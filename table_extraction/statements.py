from vietocr.tool.predictor import Predictor
from vietocr.tool.config import Cfg
from PIL import Image
import torch

def iou(box1, box2):
    """
    intersection over union of 2 boxes
    IoU = intersection(box1, box2) / (box1 + box2 - intersection)
    """
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter = abs(max((x2-x1), 0)*max((y2-y1), 0))
    if inter == 0:
        return 0

    box1_area = abs((box1[2]-box1[0])*(box1[3]-box1[1]))
    box2_area = abs((box2[2]-box2[0])*(box2[3]-box2[1]))

    return inter/(box1_area+box2_area-inter)

def get_coordinate(box):
    """
    Get coordinate of top-left and right-bottom of a box
    """
    return int(box[0][0]), int(box[0][1]), int(box[2][0]), int(box[2][1])

class Statements():
    def __init__(self, image, ocr, vietocr_weight):
        self.image = image
        self.image_height = self.image.shape[0]
        self.image_width = self.image.shape[1]
        self.ocr = ocr

        config = Cfg.load_config_from_name('vgg_transformer')
        config['weights'] = vietocr_weight
        config['device'] = 'cuda' if torch.cuda.is_available() else 'cpu'
        config['cnn']['pretrained']=False
        config['predictor']['beamsearch']=False
        self.detector = Predictor(config)

    def check_same_line(self, box1, box2):
        box1[0] = 0
        box1[2] = self.image_width

        box2[0] = 0
        box2[2] = self.image_width

        if iou(box1, box2) > 0.5:
            return 1

        return 0

    def __call__(self):
        output = self.ocr.ocr(self.image)[0]
        boxes = [line[0] for line in output]
        texts = ""
        b = [0, 0, 0, 0]
        metadata={}
        for box, idx in zip(boxes, range(len(boxes))):
            x1, y1, x2, y2 = get_coordinate(box)
            cropped_img = self.image[max(0, y1-10): min(y2+10, self.image_height), max(0, x1-10): min(x2+10, self.image_width)]
            cropped_img = Image.fromarray(cropped_img)
            text = self.detector.predict(cropped_img)
            
            if self.check_same_line(b, [x1, y1, x2, y2])==1:
                texts+="\t\t\t\t"+text
            else: 
                texts+="\n"+text

            b = [x1, y1, x2, y2]
            metadata[idx]={}
            metadata[idx]['coordinate'] = [x1, y1, x2, y2]
            metadata[idx]['text'] = text

        return texts[1:], metadata

