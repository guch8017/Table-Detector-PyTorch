import cv2
import numpy as np
import torch
from skimage import measure
from scipy.spatial import distance as dist

try:
    from .utils import letterbox_image, solve
except ImportError:
    from utils import letterbox_image, solve


def _order_points(pts):
    """
    根据x坐标对点进行排序
    原文CSDN打不开了
    """
    x_sorted = pts[np.argsort(pts[:, 0]), :]

    left_most = x_sorted[:2, :]
    right_most = x_sorted[2:, :]
    left_most = left_most[np.argsort(left_most[:, 1]), :]
    (tl, bl) = left_most

    distance = dist.cdist(tl[np.newaxis], right_most, "euclidean")[0]
    (br, tr) = right_most[np.argsort(distance)[::-1], :]

    return np.array([tl, tr, br, bl], dtype="float32")


def image_location_sort_box(box):
    x1, y1, x2, y2, x3, y3, x4, y4 = box[:8]
    pts = (x1, y1), (x2, y2), (x3, y3), (x4, y4)
    pts = np.array(pts, dtype=np.float32)
    (x1, y1), (x2, y2), (x3, y3), (x4, y4) = _order_points(pts)
    return [x1, y1, x2, y2, x3, y3, x4, y4]


def min_area_rect(coords):
    """
    多边形外接矩形
    """
    rect = cv2.minAreaRect(coords[:, ::-1])
    box = cv2.boxPoints(rect)
    box = box.reshape((8,)).tolist()
    box = image_location_sort_box(box)
    x1, y1, x2, y2, x3, y3, x4, y4 = box
    degree, w, h, cx, cy = solve(box)
    if w < h:
        x_min = (x1 + x2) / 2
        x_max = (x3 + x4) / 2
        y_min = (y1 + y2) / 2
        y_max = (y3 + y4) / 2

    else:
        x_min = (x1 + x4) / 2
        x_max = (x2 + x3) / 2
        y_min = (y1 + y4) / 2
        y_max = (y2 + y3) / 2
    return [x_min, y_min, x_max, y_max]


def get_table_line(bin_img, axis=0, line_w=10):
    """
    获取表格线
    xis=0 横线, axis=1 竖线
    :param bin_img:
    :param axis: 0 横线; 1 竖线
    :param line_w: 直线长度阈值
    :return:
    """
    labels = measure.label(bin_img > 0, connectivity=2)  # 8连通区域标记
    regions = measure.regionprops(labels)
    if axis == 1:
        line_boxes = [min_area_rect(line.coords) for line in regions if line.bbox[2] - line.bbox[0] > line_w]
    else:
        line_boxes = [min_area_rect(line.coords) for line in regions if line.bbox[3] - line.bbox[1] > line_w]
    return line_boxes


def get_distance(p1, p2):
    return np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


def adjust_lines(rows_lines, cols_lines, alph=50):
    """
    调整line
    :param rows_lines:
    :param cols_lines:
    :param alph:
    :return:
    """
    n_row = len(rows_lines)
    n_col = len(cols_lines)
    new_rows_lines = []
    new_cols_lines = []
    for i in range(n_row):
        x1, y1, x2, y2 = rows_lines[i]
        cx1, cy1 = (x1 + x2) / 2, (y1 + y2) / 2
        for j in range(n_row):
            if i != j:
                x3, y3, x4, y4 = rows_lines[j]
                cx2, cy2 = (x3 + x4) / 2, (y3 + y4) / 2
                if (x3 < cx1 < x4 or y3 < cy1 < y4) or (x1 < cx2 < x2 or y1 < cy2 < y2):
                    continue
                else:
                    r = get_distance((x1, y1), (x3, y3))
                    if r < alph:
                        new_rows_lines.append([x1, y1, x3, y3])
                    r = get_distance((x1, y1), (x4, y4))
                    if r < alph:
                        new_rows_lines.append([x1, y1, x4, y4])

                    r = get_distance((x2, y2), (x3, y3))
                    if r < alph:
                        new_rows_lines.append([x2, y2, x3, y3])
                    r = get_distance((x2, y2), (x4, y4))
                    if r < alph:
                        new_rows_lines.append([x2, y2, x4, y4])

    for i in range(n_col):
        x1, y1, x2, y2 = cols_lines[i]
        cx1, cy1 = (x1 + x2) / 2, (y1 + y2) / 2
        for j in range(n_col):
            if i != j:
                x3, y3, x4, y4 = cols_lines[j]
                cx2, cy2 = (x3 + x4) / 2, (y3 + y4) / 2
                if (x3 < cx1 < x4 or y3 < cy1 < y4) or (x1 < cx2 < x2 or y1 < cy2 < y2):
                    continue
                else:
                    r = get_distance((x1, y1), (x3, y3))
                    if r < alph:
                        new_cols_lines.append([x1, y1, x3, y3])
                    r = get_distance((x1, y1), (x4, y4))
                    if r < alph:
                        new_cols_lines.append([x1, y1, x4, y4])

                    r = get_distance((x2, y2), (x3, y3))
                    if r < alph:
                        new_cols_lines.append([x2, y2, x3, y3])
                    r = get_distance((x2, y2), (x4, y4))
                    if r < alph:
                        new_cols_lines.append([x2, y2, x4, y4])

    return new_rows_lines, new_cols_lines


def line_to_line(points1, points2, alpha=10):
    """
    线段之间的距离
    """

    def fit_line(p1, p2):
        """
        返回直线一般方程 AX+BY+C=0
        """
        x1_, y1_ = p1
        x2_, y2_ = p2
        a = y2_ - y1_
        b = x1_ - x2_
        c = x2_ * y1_ - x1_ * y2_
        return a, b, c

    def point_line_cor(ptr, a, b, c):
        """
        判断点与之间的位置关系
        :param ptr:
        :param a:
        :param b:
        :param c:
        :return:
        """
        x_, y_ = ptr
        r = a * x_ + b * y_ + c
        return r

    x1, y1, x2, y2 = points1
    ox1, oy1, ox2, oy2 = points2
    a1, b1, c1 = fit_line((x1, y1), (x2, y2))
    a2, b2, c2 = fit_line((ox1, oy1), (ox2, oy2))
    flag1 = point_line_cor([x1, y1], a2, b2, c2)
    flag2 = point_line_cor([x2, y2], a2, b2, c2)

    if (flag1 > 0 and flag2 > 0) or (flag1 < 0 and flag2 < 0):

        x = (b1 * c2 - b2 * c1) / (a1 * b2 - a2 * b1)
        y = (a2 * c1 - a1 * c2) / (a1 * b2 - a2 * b1)
        p = (x, y)
        r0 = get_distance(p, (x1, y1))
        r1 = get_distance(p, (x2, y2))

        if min(r0, r1) < alpha:

            if r0 < r1:
                points1 = [p[0], p[1], x2, y2]
            else:
                points1 = [x1, y1, p[0], p[1]]

    return points1


def table_line(img, model, device, size=(512, 512), h_prob=0.5, v_prob=0.5, row=50, col=30, alph=15):
    size_w, size_h = size
    input_blob, fx, fy = letterbox_image(img[..., ::-1], (size_w, size_h))
    input_blob = torch.from_numpy(input_blob / 255.0).permute(2, 0, 1).unsqueeze(0).float()
    pred = model(input_blob.to(device)).to('cpu').detach().permute(0, 2, 3, 1).numpy()
    pred = pred[0]
    v_pred = pred[..., 1] > v_prob  # 竖线
    h_pred = pred[..., 0] > h_prob  # 横线
    v_pred = v_pred.astype(int)
    h_pred = h_pred.astype(int)
    col_boxes = get_table_line(v_pred, axis=1, line_w=col)
    row_boxes = get_table_line(h_pred, axis=0, line_w=row)
    c_col_box = []
    c_row_box = []
    if len(row_boxes) > 0:
        row_boxes = np.array(row_boxes)
        row_boxes[:, [0, 2]] = row_boxes[:, [0, 2]] / fx
        row_boxes[:, [1, 3]] = row_boxes[:, [1, 3]] / fy
        x_min = row_boxes[:, [0, 2]].min()
        x_max = row_boxes[:, [0, 2]].max()
        y_min = row_boxes[:, [1, 3]].min()
        y_max = row_boxes[:, [1, 3]].max()
        c_col_box = [[x_min, y_min, x_min, y_max], [x_max, y_min, x_max, y_max]]
        row_boxes = row_boxes.tolist()

    if len(col_boxes) > 0:
        col_boxes = np.array(col_boxes)
        col_boxes[:, [0, 2]] = col_boxes[:, [0, 2]] / fx
        col_boxes[:, [1, 3]] = col_boxes[:, [1, 3]] / fy

        x_min = col_boxes[:, [0, 2]].min()
        x_max = col_boxes[:, [0, 2]].max()
        y_min = col_boxes[:, [1, 3]].min()
        y_max = col_boxes[:, [1, 3]].max()
        col_boxes = col_boxes.tolist()
        c_row_box = [[x_min, y_min, x_max, y_min], [x_min, y_max, x_max, y_max]]

    row_boxes += c_row_box
    col_boxes += c_col_box

    r_boxes_row_, r_boxes_col_ = adjust_lines(row_boxes, col_boxes, alph=alph)
    row_boxes += r_boxes_row_
    col_boxes += r_boxes_col_
    n_row = len(row_boxes)
    n_col = len(col_boxes)
    for i in range(n_row):
        for j in range(n_col):
            row_boxes[i] = line_to_line(row_boxes[i], col_boxes[j], 10)
            col_boxes[j] = line_to_line(col_boxes[j], row_boxes[i], 10)

    return row_boxes, col_boxes
