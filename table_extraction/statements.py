from vietocr.tool.predictor import Predictor
from vietocr.tool.config import Cfg
from PIL import Image
import torch

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

    def __call__(self):
        output = self.ocr.ocr(self.image)[0]
        boxes = [line[0] for line in output]
        texts = ""
        for box in boxes:
            x1, y1, x2, y2 = get_coordinate(box)
            cropped_img = self.image[max(0, y1-10): min(y2+10, self.image_height), max(0, x1-10): min(x2+10, self.image_width)]
            cropped_img = Image.fromarray(cropped_img)
            texts+= self.detector.predict(cropped_img)+"\n"
        return texts

