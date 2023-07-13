import cv2
from bs4 import BeautifulSoup
import xlsxwriter
import pandas as pd

def postprocess_table(excel_path, image_path, table_engine): 
    img = cv2.imread(image_path)
    result = table_engine(img)

    html = result[0]['res']['html']
    soup = BeautifulSoup(html, 'html.parser')
    head = soup.find('tr')
    print(head)
    df = pd.read_csv(excel_path)
    workbook = xlsxwriter.Workbook(excel_path)
    worksheet = workbook.add_worksheet()
    merge_format = workbook.add_format(
        {
            "align": "center",
            "valign": "vcenter",
        }
    )

    for row_num, row_data in df.iterrows():
        for col_num, value in enumerate(row_data):
            try:
                worksheet.write(row_num, col_num, value)
            except:
                pass

    idx = 1
    for td in head.find_all('td'):
        if len(td.attrs) == 0 :
            idx+=1
        else: 
            for key in td.attrs:
                if key == "rowspan":
                    try: 
                        worksheet.merge_range(0, idx, 1, idx, df.iloc[0, idx] + "\n" + df.iloc[1, idx], merge_format)
                        idx+=1
                    except:
                        idx+=1
                
                if key == "colspan":
                    try: 
                        worksheet.merge_range(0, idx, 0, idx+int(td[key])-1, df.iloc[0, idx], merge_format)
                        idx+=2
                    except:
                        idx+=2
        

    workbook.close()