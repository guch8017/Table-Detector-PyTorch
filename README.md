# Table-Detector-PyTorch

A TableDetect model implemented in PyTorch.

Original Tensorflow version by [@chineseocr](https://github.com/chineseocr): [table-detect](https://github.com/chineseocr/table-detect)

# Usage

### 1. Prepare environment

First, you need to prepare a python3 with PyTorch installed. Follow the instructions on [PyTorch](https://pytorch.org/) to install PyTorch. If you want to use GPU for inference, you need to install the CUDA version.

Then, clone this repository and install dependencies:

```bash
git clone --depth=1 
cd Table-Detector-PyTorch
pip install -r requirements.txt
```

### 2. Download model checkpoint
* Baidu Net-disk: [table-detect.pth](https://pan.baidu.com/s/12gXSFcADXhlYK6vymlcbuQ) Password: h2o6 

### 3. Detect a table structure

```python
import cv2
from extractor import TableExtractor

image = cv2.imread('PATH/TO/YOUR/TABLE/IMAGE.jpg')
extractor = TableExtractor(r'PATH/TO/MODEL_CHECKPOINT.pth', 'cpu')  # If you want to use gpu, then modify 'cpu' to 'cuda'
extractor.set_image(image)  # Set the image to be detected and perform detection
builder = extractor.get_builder()
# Save the table-cell outline as an image
cv2.imwrite('test.png', builder.to_image())
# Save the table-cell outline as a excel(.xls) file
builder.to_excel('test.xls')
```

# 4. OCR
If you want to rebuild the table with text, you can use the following code to perform OCR on each table-cell.
The engine used in demo is [PaddleOCR with PaddleServing](https://github.com/PaddlePaddle/PaddleOCR/blob/release/2.6/deploy/pdserving/README.md) running on localhost and default port 9292. 
If you want to use other ocr engine, just implement the `OCREngine` class in `ocr.py`.

```python
import cv2
from extractor import TableExtractor
from ocr import PaddleOCREngine

image = cv2.imread('PATH/TO/YOUR/TABLE/IMAGE.jpg')
engine = PaddleOCREngine()
extract = TableExtractor(model_path='table-line.pth', model_device='cpu')
extract.set_image(image)
builder = extract.get_builder()
builder.set_ocr_engine(engine)
builder.do_ocr(process=32)
builder.to_excel('test.xls')
```

# License

MIT License