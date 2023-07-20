import pandas as pd
from sqlalchemy import create_engine
import json

class OCRDatabase():

	def __init__(self, mode, json_path):
	# Credentials to database connection
		self.hostname="localhost"
		self.dbname="OCR"
		self.uname="root"
		self.pwd="Anhviet140220"
		self.mode = mode
		self.path = json_path
		f = open(json_path)
		self.data = json.load(f)

	def __call__(self):
		pdf = self.path.split('/')[1]
		img = self.path.split('/')[-1]
		img = img.split('.')[0]
		pdf_data = []
		if self.mode == "Table":
			for row in self.data.keys():
				for col in self.data[row].keys():
					if(len(self.data[row][col])==0):
						continue
					coor = self.data[row][col]['coordinate']
					text = self.data[row][col]['text']
					pdf_data.append([pdf, img, coor[0], coor[1], coor[2], coor[3], text])
		else: 
			for row in self.data.keys():
				coor = self.data[row]['coordinate']
				text = self.data[row]['text']
				pdf_data.append([pdf, img, coor[0], coor[1], coor[2], coor[3], text])

		df = pd.DataFrame(data=pdf_data,columns=['PDF_Id', 'Page_Id', 'x1', 'y1', 'x2', 'y2', 'Texts'])

		# Create SQLAlchemy engine to connect to MySQL Database
		engine = create_engine("mysql+pymysql://{user}:{pw}@{host}/{db}"
						.format(host=self.hostname, db=self.dbname, user=self.uname, pw=self.pwd))
		myconn = engine.connect()
		# Convert dataframe to sql table                                   
		df.to_sql('Pdf', myconn, index=False, if_exists='append')