### PP-STRUCTURE ENVIROMENT PREPARATION

# Install torch
pip install torch==1.8.1+cpu torchvision==0.9.1+cpu torchaudio==0.8.1 -f https://download.pytorch.org/whl/torch_stable.html
# Install PaddlePaddle for CPU
python3 -m pip install paddlepaddle -i https://mirror.baidu.com/pypi/simple
git clone --branch v2.5.0 https://github.com/PaddlePaddle/PaddleOCR.git
pip install paddleocr "paddleocr>=2.0.1"

# Install necessary libaries
pip install -r requirements.txt