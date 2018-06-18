from sympy.geometry import point
import numpy as np

class wells:
    def init(self, x, y, w, h):
        coord = point.Point2D(float(x), float(y))
        width = w
        high = h

class cand():
    def __init__(self, name, x, y, w, h):
        self.cands = []
        self.cands.append((point.Point2D(x, y), point.Point2D(x - w, y + h)))
        self.cands.append((point.Point2D(x, y), point.Point2D(x - w, y - h)))
        self.cands.append((point.Point2D(x, y), point.Point2D(x + w, y + h)))
        self.cands.append((point.Point2D(x, y), point.Point2D(x + w, y - h)))
        self.cands.append((point.Point2D(x - w / 2, y), point.Point2D(x + w / 2, y + h)))
        self.cands.append((point.Point2D(x, y + h / 2), point.Point2D(x + w, y - h / 2)))
        self.cands.append((point.Point2D(x + w / 2, y), point.Point2D(x - w / 2, y - h)))
        self.cands.append((point.Point2D(x, y - h / 2), point.Point2D(x - w, y + h / 2)))
        self.name = name
        self.center = point.Point2D(x, y)
        self.trouble = np.zeros((4,1))