### ENVIROMENT PREPARATION

# Install torch
pip install torch==1.8.1+cpu torchvision==0.9.1+cpu torchaudio==0.8.1 -f https://download.pytorch.org/whl/torch_stable.html

# Install PaddlePaddle for CPU
python3 -m pip install paddlepaddle -i https://mirror.baidu.com/pypi/simple
pip install paddleocr "paddleocr>=2.0.1"
pip install protobuf==3.20.0
pip install pybind11

# Install tesseract for ubuntu
sudo apt install tesseract-ocr

# Install necessary libaries
pip install -r requirements.txt

### RUN CODE
python main.py --dir path_to_pdf_file

Kết quả sẽ được trả về trong thư mục result/pdf_file_name/