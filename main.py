import argparse
from pdf2image import convert_from_path
import layoutparser as lp
from table_extraction.borderless import BorderlessTable
from util.pages_analysis import PagesAnalysis
from paddleocr import PaddleOCR
import cv2

def main(args):
    images = convert_from_path(args.pdf)
    for i in range(len(images)):
        images[i].save('pages/page'+str(i)+'.jpg', 'JPEG')
    ocr = PaddleOCR(lang='en')
    # Tesseract Configuration parameters
    # oem --> OCR engine mode = 3 >> Legacy + LSTM mode only (LSTM neutral net mode works the best)
    # psm --> page segmentation mode = 6 >> Assume as single uniform block of text (How a page of text can be analyzed)
    tesseract_config = r'--oem 3 --psm 6'
    pages = PagesAnalysis(images, ocr, tesseract_config)
    balance, income_statement, cash_flow_statement = pages()

    model = lp.PaddleDetectionLayoutModel(config_path="lp://PubLayNet/ppyolov2_r50vd_dcn_365e_publaynet/config",
                                threshold=0.5,
                                label_map={0: "Text", 1: "Title", 2: "List", 3:"Table", 4:"Figure"},
                                enforce_cpu=False,
                                enable_mkldnn=True)

    for reporting in [balance, income_statement, cash_flow_statement]:
        for i in range(len(reporting)):
            image = cv2.imread('pages/page'+str(reporting[i])+'.jpg')
            image = image[..., ::-1]
            layout = model.detect(image)
            for l in layout:
                if l.type == 'Table':
                    x1 = int(l.block.x_1)
                    y1 = int(l.block.y_1)
                    x2 = int(l.block.x_2)
                    y2 = int(l.block.y_2)

            im = image[y1:y2, x1:x2]
            extract_information = BorderlessTable(im, ocr, tesseract_config)
            df = extract_information()
            print(df)

if __name__== "__main__":
    parser = argparse.ArgumentParser(description='PyTorch margin penalty loss  training')
    parser.add_argument('--pdf', type=str, help='path to pdf file to extract')
    args_ = parser.parse_args()
    main(args_)
