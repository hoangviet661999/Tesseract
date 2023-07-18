import pandas as pd
from sqlalchemy import create_engine
import json

# Credentials to database connection
hostname="localhost"
dbname="OCR"
uname="root"
pwd="Anhviet140220"

f = open('result/8_3__bctc_kiem_toan_2020_RYWJ/balance/image5.json')
data = json.load(f)

pdf_data = []
for row in data.keys():
	for col in data[row].keys():
		coor = data[row][col]['coordinate']
		text = data[row][col]['text']
		pdf_data.append(['8', '5', coor[0], coor[1], coor[2], coor[3], text])

df = pd.DataFrame(data=pdf_data,columns=['PDF_Id', 'Page_Id', 'x1', 'y1', 'x2', 'y2', 'Texts'])

# Create SQLAlchemy engine to connect to MySQL Database
engine = create_engine("mysql+pymysql://{user}:{pw}@{host}/{db}"
				.format(host=hostname, db=dbname, user=uname, pw=pwd))
myconn = engine.connect()
# Convert dataframe to sql table                                   
df.to_sql('pdf', myconn, index=False, if_exists='append')