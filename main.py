import argparse
from pdf2image import convert_from_path
from table_extraction.borderless import BorderlessTable
from paddleocr import PaddleOCR
import os
import torch
import numpy as np
from util.pdf_analysis import PdfAnalysis
import time
from table_extraction.statements import Statements
from util.preprocess import prepocess_image

def take_page(images, start, end, tab_model, ocr, tesseract_config, vietocr_weight, folder, name):
    for i in range(start, end):
        #read images and preprocessing
        image = np.array(images[i])
        image = prepocess_image(image)
        #feed images to table detection model
        results = tab_model(image)
        boxes = results.xyxy[0]
        boxes = boxes[0]
        x1 = int(boxes[0])
        y1 = int(boxes[1])
        x2 = int(boxes[2])
        y2 = int(boxes[3])
        im = image[max(0, y1-30):min(y2+30, image.shape[0]), max(0, x1-30):min(x2+30, image.shape[1])]
        extract_information = BorderlessTable(im, ocr, tesseract_config, vietocr_weight)
        df = extract_information()
        df.to_csv('result/{}/{}_{}.csv'.format(folder, name, str(i)))

def parse_pdf(dir, sig_model, tab_model, ocr, tesseract_config, vietocr_weight, folder):
    images = convert_from_path(dir, dpi=400)

    pdf = PdfAnalysis(images, tab_model, sig_model)
    pages = pdf()

    take_page(images, pages[0], pages[1]+1, tab_model, ocr, tesseract_config, vietocr_weight, folder, name="balance") #balance sheet
    take_page(images, pages[1]+1, pages[2]+1, tab_model, ocr, tesseract_config, vietocr_weight, folder, name="income") #income statement
    take_page(images, pages[2]+1, pages[3]+1, tab_model, ocr, tesseract_config, vietocr_weight, folder, name="cashflow") #cash flow statement

    for i in range(pages[3]+1, len(images)):
        image = np.array(images[i])
        image = prepocess_image(image)
        results = tab_model(image)
        boxes = results.xyxy[0]
        if len(boxes)>0:
            for box, idx in zip(boxes, range(len(boxes))):
                if box[4] > 0.5: 
                    x1 = int(box[0])
                    y1 = int(box[1])
                    x2 = int(box[2])
                    y2 = int(box[3])
                    im = image[max(0, y1-30):min(y2+30, image.shape[0]), max(0, x1-30):min(x2+30, image.shape[1])]
                    extract_information = BorderlessTable(im, ocr, tesseract_config, vietocr_weight)
                    df = extract_information()
                    df.to_csv('result/{}/page_{}_{}.csv'.format(folder, str(i), str(idx)))

        if len(boxes)>0:
            for box in boxes:
                if box[4] > 0.5: 
                    x1 = int(box[0])
                    y1 = int(box[1])
                    x2 = int(box[2])
                    y2 = int(box[3])
                    image[max(0, y1-30):min(y2+30, image.shape[0]), max(0, x1-30):min(x2+30, image.shape[1])] = 255

        stm = Statements(image, ocr, vietocr_weight)
        text = stm()
        text_file = open("result/{}/page{}.txt".format(folder, str(i)), "w")
        text_file.write(text)
        text_file.close()
    
def main(args):
    start = time.time()
    sig_model = torch.hub.load('ultralytics/yolov5', 'custom', path='sig_best.pt')
    tab_model = torch.hub.load('ultralytics/yolov5', 'custom', path='tab_best.pt')
    ocr = PaddleOCR(lang='en')
    vietocr_weight = 'transformerocr.pth'
    # Tesseract Configuration parameters
    # oem --> OCR engine mode = 3 >> Legacy + LSTM mode only (LSTM neutral net mode works the best)
    # psm --> page segmentation mode = 6 >> Assume as single uniform block of text (How a page of text can be analyzed)
    tesseract_config = r'--oem 3 --psm 6'
    if os.path.isfile(args.dir):
        folder = args.dir.split('.')[0]
        if not os.path.exists('result/{}'.format(folder)):
            os.mkdir('result/{}'.format(folder))
        parse_pdf(args.dir, sig_model, tab_model, ocr, tesseract_config, vietocr_weight, folder)

    else:
        for file in os.listdir(args.dir):
            folder = file.split('.')[0]
            if not os.path.exists('result/{}'.format(folder)):
                os.mkdir('result/{}'.format(folder))
            pdf_path = os.path.join(args.dir, file)
            parse_pdf(pdf_path, sig_model, tab_model, ocr, tesseract_config, vietocr_weight, folder)
    end = time.time()
    print(end-start)


if __name__== "__main__":
    parser = argparse.ArgumentParser(description='PyTorch margin penalty loss  training')
    parser.add_argument('--dir', type=str, help='path to folder or pdf file to extract')
    args_ = parser.parse_args()
    main(args_)