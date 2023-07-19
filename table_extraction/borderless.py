import numpy as np
import tensorflow as tf
import pandas as pd
from vietocr.tool.predictor import Predictor
from vietocr.tool.config import Cfg
from PIL import Image
import torch

def redistributed(boxes, texts, image_width):
    new_boxes = [boxes[0]]
    new_texts = [texts[0]]
    for idx in range(1, len(texts)):
        if len(texts[idx]) == 0:
            new_boxes.append(boxes[idx])
            new_texts.append(texts[idx])
        else:
            if texts[idx][0].islower() or \
                ((texts[idx][0] == "(" or texts[idx][0] == "{" or texts[idx][0] == "[") and texts[idx][1:5]!="Tăng" and boxes[idx][0][0] < image_width/6):
                b = -1
                r = 1e9
                for i in range(max(0,len(new_boxes)-10), len(new_boxes)):
                    x1, y1, x2, y2 = get_coordinate(boxes[idx])
                    x3, y3, x4, y4 = get_coordinate(new_boxes[i])
                    if  abs(y1-y4) + abs(x1-x3) < r:
                        r = abs(y1-y4) + abs(x1-x3)
                        b = i
                new_texts.append(new_texts[b]+"\n"+texts[idx])
                new_texts.pop(b)

                x1, y1, x2, y2 = get_coordinate(boxes[idx])
                x3, y3, x4, y4 = get_coordinate(new_boxes[b])

                x1_res = min(x1, x2, x3, x4)
                x2_res = max(x1, x2, x3, x4)
                y1_res = min(y1, y2, y3, y4)
                y2_res = max(y1, y2, y3, y4)

                new_boxes.append([[x1_res, y1_res], [x2_res, y1_res], [x2_res, y2_res], [x1_res, y2_res]])
                new_boxes.pop(b)

            else:
                new_boxes.append(boxes[idx])
                new_texts.append(texts[idx])

    return new_boxes, new_texts

def intersection(box1, box2):
    """
    intersection of 2 boxes
    """
    return [box2[0], box1[1], box2[2], box1[3]]

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

    return inter 

def get_coordinate(box):
    """
    Get coordinate of top-left and right-bottom of a box
    """
    return int(box[0][0]), int(box[0][1]), int(box[2][0]), int(box[2][1])

class BorderlessTable():
    """
    Automatically reading information from a table image with no border,
    reconstructing table and writing information into a CSV file 

    Args:
        image(Mat): image of a table with no border
    Returns:
        table(Dataframe): Dataframe with information and construction's table image
    """
    def __init__(self, image, ocr, tesseract_config, vietocr_weight):
        self.image = image
        self.ocr = ocr
        self.image_height = image.shape[0]
        self.image_width = image.shape[1]
        self.tesseract_config = tesseract_config
        config = Cfg.load_config_from_name('vgg_transformer')
        config['weights'] = vietocr_weight
        config['device'] = 'cuda' if torch.cuda.is_available() else 'cpu'
        config['cnn']['pretrained']=False
        config['predictor']['beamsearch']=False
        self.detector = Predictor(config)

    def __call__(self):
        #ocr image to get texts's location
        output = self.ocr.ocr(self.image)[0]
        boxes = [line[0] for line in output]
        texts = []
        #for each text's location, we recognize text
        for box in boxes:
            x1, y1, x2, y2 = get_coordinate(box)
            cropped_img = self.image[max(0, y1-10): min(y2+10, self.image_height), max(0, x1-10): min(x2+10, self.image_width)]
            # texts.append(pytesseract.image_to_string(cropped_img, config=self.tesseract_config, lang = 'vie')[:-2])
            cropped_img = Image.fromarray(cropped_img)
            texts.append(self.detector.predict(cropped_img))

        boxes, texts = redistributed(boxes, texts, self.image_width)
        prob = [int(box[2][1])/self.image_height for box in boxes]
        h_prob = [(int(box[2][1])-int(box[0][1])) for box in boxes]

        #expand the box horizontally and vertically 
        horiz_boxes = []
        vert_boxes = []
        for box in boxes:
            x1, y1, x2, y2 = get_coordinate(box)
            horiz_boxes.append([0, y1, self.image_width, y2])
            vert_boxes.append([x1, 0, x2, self.image_height])

        #Selects a single box out of many overlapping boxes using non-max-suppression with iou_threshold=0.1
        horiz_out = tf.image.non_max_suppression(
            horiz_boxes,
            h_prob,
            max_output_size=1000,
            iou_threshold=0.3,
            score_threshold=float('-inf'),
            name=None
        )
        horiz_line = np.sort(np.array(horiz_out))

        vert_out = tf.image.non_max_suppression(
            vert_boxes,
            prob,
            max_output_size=1000,
            iou_threshold=0.0001,
            score_threshold=float('-inf'),
            name=None
        )
        vert_line = np.sort(np.array(vert_out))

        #table reconstruction with intersection of horiz_boxes and vert_boxes
        table = [["" for _ in range(len(vert_line))] for _ in range(len(horiz_line))]

        unordered_boxes = []
        for i in vert_line:
            unordered_boxes.append(vert_boxes[i][0])
        ordered_boxes = np.argsort(unordered_boxes)

        metadata = {}
        for i in range(len(horiz_line)):
            data = {}
            for j in range(len(vert_line)):
                data[j] = {}
                resultant = intersection(horiz_boxes[horiz_line[i]], vert_boxes[vert_line[ordered_boxes[j]]])
                for b in range(len(boxes)):
                    the_box = [boxes[b][0][0], boxes[b][0][1], boxes[b][2][0], boxes[b][2][1]]
                    if(iou(resultant, the_box)>0.2*((boxes[b][2][0]-boxes[b][0][0])*(boxes[b][2][1]-boxes[b][0][1]))):
                        if table[i][j] != "":
                            if j <=1: 
                                table[i][j] = table[i][j] + " " + texts[b]
                            else:
                                table[i][j] = table[i][j] + "\n" + texts[b]
                        else:
                            table[i][j] = texts[b]
                        if 'coordinate' in data[j].keys():
                            x1, y1, x2, y2 = data[j]['coordinate'][0], data[j]['coordinate'][1], data[j]['coordinate'][2], data[j]['coordinate'][3]
                            x1_res = min(x1, x2, boxes[b][0][0], boxes[b][2][0])
                            x2_res = max(x1, x2, boxes[b][0][0], boxes[b][2][0])
                            y1_res = min(y1, y2, boxes[b][0][1], boxes[b][2][1])
                            y2_res = max(y1, y2, boxes[b][0][1], boxes[b][2][1])
                            data[j]['coordinate'] = [x1_res, y1_res, x2_res, y2_res]
                        else: 
                            data[j]['coordinate'] = [boxes[b][0][0], boxes[b][0][1], boxes[b][2][0], boxes[b][2][1]]
                        data[j]['text'] = table[i][j]
            metadata[i] = data

        return pd.DataFrame(table), metadata