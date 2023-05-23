# Table-Detector-PyTorch

A TableDetect model implemented in PyTorch.

Original Tensorflow version by [@chineseocr](https://github.com/chineseocr): [table-detect](https://github.com/chineseocr/table-detect)

# Usage

### 1. Prepare environment

```bash
git clone --depth=1 
cd table_extractor
pip install -r requirements.txt
```

### 2. Download model checkpoint
TODO

### 3. Detect a table

```python
import cv2
from table_extractor import TableExtractor

image = cv2.imread('PATH/TO/YOUR/TABLE/IMAGE.jpg')
extractor = TableExtractor(r'PATH/TO/MODEL_CHECKPOINT.pth', 'cpu')  # If you want to use gpu, then modify 'cpu' to 'cuda'
extractor.set_image(image)  # Set the image to be detected and perform detection
builder = extractor.get_builder()
# Save the table-cell outline as an image
cv2.imwrite('test.png', builder.to_image(image.shape))
# Save the table-cell outline as a excel(.xls) file
builder.to_excel('test.xls')
```

# TODO
Table cell OCR

# License

MIT License