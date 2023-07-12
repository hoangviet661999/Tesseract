### PP-STRUCTURE ENVIROMENT PREPARATION

# Install torch
pip install torch==1.8.1+cpu torchvision==0.9.1+cpu torchaudio==0.8.1 -f https://download.pytorch.org/whl/torch_stable.html

# Install PaddlePaddle for CPU
python3 -m pip install paddlepaddle -i https://mirror.baidu.com/pypi/simple
pip install paddleocr "paddleocr>=2.0.1"

# Install tesseract
sudo apt install tesseract-ocr

# Install necessary libaries
cat requirements.txt | xargs pip install