from .utils import xy_rotate_box
import cv2
import numpy as np


class TableBuilder:
    """
    表格重建
    """

    def __init__(self, ceil_boxes, interval=10):
        """
        @:param ceil_boxes: [[x0,y0,x1,y1,x2,y2,x3,y3,x4,y4]]
        """
        self.cor = []
        self.ceil_boxes = ceil_boxes
        self.diag_boxes = [[int(x[0]), int(x[1]), int(x[4]), int(x[5])] for x in ceil_boxes]
        self.interval = interval
        self.batch()

    def batch(self):
        self.cor = []
        row_cor = self.table_line_cor(self.diag_boxes, axis='row', interval=self.interval)
        col_cor = self.table_line_cor(self.diag_boxes, axis='col', interval=self.interval)
        cor = [{'row': line[1], 'col': line[0]} for line in zip(row_cor, col_cor)]
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

        if axis == 'col':
            cor = [[edges_map_index[edges_map[line[1]]], edges_map_index[edges_map[line[3]]]] for line in lines]
        else:
            cor = [[edges_map_index[edges_map[line[0]]], edges_map_index[edges_map[line[2]]]] for line in lines]
        return cor

    def to_excel(self):
        import xlwt
        row = 0
        workbook = xlwt.Workbook()
        if len(self.cor) == 0:
            worksheet = workbook.add_sheet('table')
            worksheet.write_merge(0, 0, 0, 0, "无数据")
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

    def to_image(self, shape, color=(255, 255, 255)):
        tmp = np.zeros(shape, dtype=np.uint8)
        h, w, _ = shape

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
