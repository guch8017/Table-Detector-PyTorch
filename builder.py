import multiprocessing

try:
    from .utils import xy_rotate_box
    from .ocr import OCREngine
except ImportError:
    from utils import xy_rotate_box
    from ocr import OCREngine
import cv2
import numpy as np
from multiprocessing import Pool, Manager
from functools import partial


def _ocr(cell: dict, engine: OCREngine, image: np.ndarray):
    p1, p2 = cell['box']
    text = engine.ocr(image[p1[1]:p2[1], p1[0]:p2[0], :])
    cell['text'] = text


class TableBuilder:
    """
    表格重建
    """

    def __init__(self, ceil_boxes, image, interval=10, engine=None):
        """
        :param ceil_boxes: [[x0,y0,x1,y1,x2,y2,x3,y3,x4,y4]]
        :param image: Original image
        """
        self.image = image
        self.cor = []
        self.ceil_boxes = ceil_boxes
        self.diag_boxes = [[int(x[0]), int(x[1]), int(x[4]), int(x[5])] for x in ceil_boxes]
        self.interval = interval
        self.engine = engine
        self.batch()

    def batch(self):
        self.cor = []
        row_cor, row_ind = self.table_line_cor(self.diag_boxes, axis='row', interval=self.interval)
        col_cor, col_ind = self.table_line_cor(self.diag_boxes, axis='col', interval=self.interval)
        # row: [r0, r1], col: [c0, c1], box: [TL, BR]
        cor = [{'row': line[1], 'col': line[0],
                'box': [(row_ind[line[0][0]], col_ind[line[1][0]]), (row_ind[line[0][1]], col_ind[line[1][1]])]} for
               line in zip(row_cor, col_cor)]
        self.cor = cor

    @staticmethod
    def table_line_cor(lines, axis='col', interval=10):

        if axis == 'col':
            edges = [[line[1], line[3]] for line in lines]
        else:
            edges = [[line[0], line[2]] for line in lines]

        edges = sum(edges, [])
        edges = sorted(edges)

        n_edges = len(edges)
        edges_map = {}
        for i in range(n_edges):
            if i == 0:
                edges_map[edges[i]] = edges[i]
                continue
            else:
                if edges[i] - edges_map[edges[i - 1]] < interval:
                    edges_map[edges[i]] = edges_map[edges[i - 1]]
                else:
                    edges_map[edges[i]] = edges[i]

        edges_map_list = [[key, edges_map[key]] for key in edges_map]
        edges_map_index = [line[1] for line in edges_map_list]
        edges_map_index = list(set(edges_map_index))
        edges_map_index = {x: ind for ind, x in enumerate(sorted(edges_map_index))}
        ind2edges = {ind: x for x, ind in edges_map_index.items()}

        if axis == 'col':
            cor = [[edges_map_index[edges_map[line[1]]], edges_map_index[edges_map[line[3]]]] for line in lines]
        else:
            cor = [[edges_map_index[edges_map[line[0]]], edges_map_index[edges_map[line[2]]]] for line in lines]
        return cor, ind2edges

    def set_ocr_engine(self, engine: OCREngine):
        self.engine = engine

    def do_ocr(self, process=multiprocessing.cpu_count()):
        """
        :param process: Number of processes use to run OCR
        When process is set to 1, it will use a for-loop
        When process is larger than 1, it will use multiprocessing.Pool & Map
        """
        if self.engine is None:
            raise Exception('OCR engine is not set')
        if not isinstance(process, int) or process < 1:
            raise ValueError('Process must be a positive integer')
        print('Running ocr...')
        if process == 1:
            for cell in self.cor:
                p1, p2 = cell['box']
                text = self.engine.ocr(self.image[p1[1]:p2[1], p1[0]:p2[0], :])
                cell['text'] = text
        else:
            manager = Manager()
            cor = manager.list([manager.dict(dt) for dt in self.cor])
            with Pool() as pool:
                pool.map(partial(_ocr, engine=self.engine, image=self.image), cor)
                pool.close()
                pool.join()
            self.cor = [dict(dt) for dt in cor]
            manager.shutdown()

    def to_excel(self):
        import xlwt
        row = 0
        workbook = xlwt.Workbook()
        if len(self.cor) == 0:
            worksheet = workbook.add_sheet('table')
            worksheet.write_merge(0, 0, 0, 0, "No Data")
        else:
            worksheet = workbook.add_sheet('page')
            page_row = 0
            for line in self.cor:
                row0, row1 = line['row']
                col0, col1 = line['col']
                text = line.get('text', '')
                try:
                    page_row = max(row1 - 1, page_row)
                    worksheet.write_merge(row + row0, row + row1 - 1, col0, col1 - 1, text)
                except:
                    pass
        return workbook

    def to_image(self, color=(255, 255, 255)):
        tmp = np.zeros_like(self.image.shape)
        h, w, _ = tmp.shape

        for box in self.ceil_boxes:
            if type(box) is dict:
                x1, y1, x2, y2, x3, y3, x4, y4 = xy_rotate_box(**box)
            else:
                x1, y1, x2, y2, x3, y3, x4, y4 = box[:8]

            cv2.line(tmp, (int(x1), int(y1)), (int(x2), int(y2)), color, 1, lineType=cv2.LINE_AA)
            cv2.line(tmp, (int(x2), int(y2)), (int(x3), int(y3)), color, 1, lineType=cv2.LINE_AA)
            cv2.line(tmp, (int(x3), int(y3)), (int(x4), int(y4)), color, 1, lineType=cv2.LINE_AA)
            cv2.line(tmp, (int(x4), int(y4)), (int(x1), int(y1)), color, 1, lineType=cv2.LINE_AA)

        return tmp
