import pytesseract
import numpy as np
import tensorflow as tf
import pandas as pd

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

    return inter / float(box1_area+box2_area-inter)

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
    def __init__(self, image, ocr, tesseract_config):
        self.image = image
        self.ocr = ocr
        self.image_height = image.shape[0]
        self.image_width = image.shape[1]
        self.tesseract_config = tesseract_config

    def __call__(self):
        #ocr image to get texts's location
        output = self.ocr.ocr(self.image)[0]
        boxes = [line[0] for line in output]
        texts = []
        prob = [line[1][1] for line in output]

        #for each text's location, we recognize text
        for box in boxes:
            x1, y1, x2, y2, = get_coordinate(box)
            cropped_img = self.image[y1: y2, x1: x2]
            texts.append(pytesseract.image_to_string(cropped_img, config=self.tesseract_config, lang = 'vie')[:-2])

        #expand the box horizontally and vertically 
        horiz_boxes = []
        vert_boxes = []
        for box in boxes:
            x1, y1, x2, y2, = get_coordinate(box)
            horiz_boxes.append([0, y1, self.image_width, y2])
            vert_boxes.append([x1, 0, x2, self.image_height])

        #Selects a single box out of many overlapping boxes using non-max-suppression with iou_threshold=0.1
        horiz_out = tf.image.non_max_suppression(
            horiz_boxes,
            prob,
            max_output_size=1000,
            iou_threshold=0.1,
            score_threshold=float('-inf'),
            name=None
        )
        horiz_line = np.sort(np.array(horiz_out))

        vert_out = tf.image.non_max_suppression(
            vert_boxes,
            prob,
            max_output_size=1000,
            iou_threshold=0.1,
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

        for i in range(len(horiz_line)):
            for j in range(len(vert_line)):
                resultant = intersection(horiz_boxes[horiz_line[i]], vert_boxes[vert_line[ordered_boxes[j]]])
                for b in range(len(boxes)):
                    the_box = [boxes[b][0][0], boxes[b][0][1], boxes[b][2][0], boxes[b][2][1]]
                    if(iou(resultant, the_box)>0.1):
                        table[i][j] = texts[b]

        return pd.DataFrame(table)