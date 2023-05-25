"""
This file contains the wrapper for the OCR API. It will be used to send the image to the OCR API and return the text
This implementation is based on PaddleOCR & PaddleServing running on localhost:9292.
You can deploy one by following https://github.com/PaddlePaddle/PaddleOCR/blob/release/2.6/deploy/pdserving/README.md
You can use your own OCR engine like by implementing the OCR method.
"""
from abc import ABC
import numpy as np
import requests
import base64
import json
import cv2


class OCREngine(ABC):
    def __init__(self, **kwargs):
        """
        Init your ocr engine here
        """
        pass

    def ocr(self, image) -> str:
        """
        This method will be called by the pipeline to perform OCR on the image.
        :param image: np.ndarray [h,w,c]
        :return: str
        """
        raise NotImplementedError


class PaddleOCREngine(OCREngine):
    def __init__(self, api: str = "http://localhost:9292/ocr/prediction", scale: int = None, **kwargs):
        """
        Paddle OCR Engine
        :param api: The API endpoint of the OCR service
        :param scale: The scale of the image to be sent to the OCR service
        """
        super().__init__(**kwargs)
        self.api = api
        self.scale = scale
        self.header = {"Content-type": "application/json"}

    def ocr(self, image: np.ndarray) -> str:
        if self.scale is not None:
            image = cv2.resize(image, None, fx=self.scale, fy=self.scale)
        image_bytes = cv2.imencode('.jpg', image)[1]
        image_base64 = base64.b64encode(image_bytes.tostring()).decode('utf8')
        data = {"feed": [{"x": image_base64}], "fetch": ["res"]}
        r = requests.post(url=self.api, headers=self.header, data=json.dumps(data))
        result = r.json()['result']
        if isinstance(result, str):
            return ''
        else:
            return '\n'.join(result['res'])
