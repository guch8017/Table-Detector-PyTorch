import cv2
import torch
import numpy as np
from PIL import Image
from .utils import estimate_skew_angle, image_location_sort_box, xy_rotate_box, solve
from .builder import TableBuilder
from .line import table_line
from .model import LineDetector
from skimage import measure


class TableExtractor(object):
    def __init__(self, img: np.ndarray, model_path: str, model_device='cuda:0', table_line_size=(1024, 1024), eval_rotate: bool = False):
        """
        表格提取
        :param img: np.ndarray [h,w,c]
        :param table_line_size:
        :param eval_rotate: 是否进行旋转修正
        """
        # 表格检测器，此处跳过，后续用微软的DocTransformer代替
        self.child_images = []
        self.table_cell_boxes = []
        self.scores = [0]
        self.ad_boxes = [[0, 0, img.shape[1], img.shape[0]]]
        self.boxes = [[0, 0, img.shape[1], img.shape[0]]]
        self.degree = 0  # 表格旋转检测  +-5度
        self.img = img
        self.table_line_size = table_line_size
        model = LineDetector(2, use_bias=False).to(model_device)
        model.load_state_dict(torch.load(model_path, map_location=model_device))
        model.eval()
        self.device = model_device
        self.model = model
        if eval_rotate:
            self.eval_rotate()
        self.extract_table_cell()

    def eval_rotate(self):
        im = Image.fromarray(self.img)
        self.degree = estimate_skew_angle(np.array(im.convert('L')), angle_range=[-15, 15])
        self.img = np.array(
            im.rotate(self.degree, center=(im.size[0] / 2, im.size[1] / 2), expand=True, fillcolor=(255, 255, 255)))

    @staticmethod
    def draw_lines(im, bboxes, color=(0, 0, 0), lineW=3):
        """
        绘制表格框线
        """
        tmp = np.copy(im)
        c = color
        h, w = im.shape[:2]

        for box in bboxes:
            x1, y1, x2, y2 = box[:4]
            cv2.line(tmp, (int(x1), int(y1)), (int(x2), int(y2)), c, lineW, lineType=cv2.LINE_AA)

        return tmp

    @staticmethod
    def min_area_rect_box(regions, flag=True, W=0, H=0, filter_small=False, adjust_box=False):
        """
        多边形外接矩形
        """
        boxes = []
        for region in regions:
            rect = cv2.minAreaRect(region.coords[:, ::-1])

            box = cv2.boxPoints(rect)
            box = box.reshape((8,)).tolist()
            box = image_location_sort_box(box)
            x1, y1, x2, y2, x3, y3, x4, y4 = box
            angle, w, h, cx, cy = solve(box)
            if adjust_box:
                x1, y1, x2, y2, x3, y3, x4, y4 = xy_rotate_box(cx, cy, w + 5, h + 5, angle=0, degree=None)

            if w > 32 and h > 32 and flag:
                if abs(angle / np.pi * 180) < 20:
                    if filter_small and w < 10 or h < 10:
                        continue
                    boxes.append([x1, y1, x2, y2, x3, y3, x4, y4])
            else:
                if w * h < 0.5 * W * H:
                    if filter_small and w < 8 or h < 8:
                        continue
                    boxes.append([x1, y1, x2, y2, x3, y3, x4, y4])
        return boxes

    def extract_table_cell(self):
        """
        提取表格单元格
        :return:
        """
        n = len(self.ad_boxes)
        for i in range(n):
            x_min, y_min, x_max, y_max = [int(x) for x in self.ad_boxes[i]]

            child_img = self.img[y_min:y_max, x_min:x_max]
            row_boxes, col_boxes = table_line(child_img[..., ::-1], self.model, self.device, size=self.table_line_size, h_prob=0.5, v_prob=0.5)
            tmp = np.zeros(self.img.shape[:2], dtype='uint8')
            tmp = self.draw_lines(tmp, row_boxes + col_boxes, color=255, lineW=2)
            labels = measure.label(tmp < 255, connectivity=2)  # 8连通区域标记
            regions = measure.regionprops(labels)
            ceil_boxes = self.min_area_rect_box(regions, False, tmp.shape[1], tmp.shape[0], True, True)
            ceil_boxes = np.array(ceil_boxes)
            ceil_boxes[:, [0, 2, 4, 6]] += x_min
            ceil_boxes[:, [1, 3, 5, 7]] += y_min
            self.table_cell_boxes.extend(ceil_boxes)
            self.child_images.append(child_img)

    def get_builder(self):
        builder = TableBuilder(self.table_cell_boxes)
        return builder

    def table_ocr(self):
        """use ocr and match ceil"""
        pass
