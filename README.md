### PP-STRUCTURE ENVIROMENT PREPARATION

# Install PaddlePaddle for CPU
python3 -m pip install paddlepaddle -i https://mirror.baidu.com/pypi/simple
git clone --branch v2.5.0 https://github.com/PaddlePaddle/PaddleOCR.git
pip install paddleocr "paddleocr>=2.0.1"

# Install necessary libaries
pip install -r requirements.txt