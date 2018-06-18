from sympy.geometry import point
import numpy as np

class cand():
    def __init__(self, name, x1, y1, x2, y2):
        self.name = name
        self.cands = []
        w = abs(x1 - x2)
        h = abs(y2 - y1)

        self.cands.append(candidat((point.Point2D(x1, y1), point.Point2D(x1 - w, y1 + h)), []))#left top
        self.cands.append(candidat((point.Point2D(x1, y1), point.Point2D(x1 - w, y1 - h)), []))#left bottom
        self.cands.append(candidat((point.Point2D(x1, y1), point.Point2D(x1 + w, y1 + h)), []))#right top
        self.cands.append(candidat((point.Point2D(x1, y1), point.Point2D(x1 + w, y1 - h)), []))#right bottom
        #cands.append(candidat((point.Point2D(x1 - w/2, y1), point.Point2D(x1 + w/2, y1 + h)), 0))#top center
        #cands.append(candidat((point.Point2D(x1, y1 + h/2), point.Point2D(x1 + w, y1 - h/2)), 0))  #left center
        #cands.append(candidat((point.Point2D(x1 + w/2, y1), point.Point2D(x1 - w/2, y1 - h)), 0))  #bottom center
        #cands.append(candidat((point.Point2D(x1, y1 - h/2), point.Point2D(x1 - w, y1 + h/2)), 0))  #right center
        self.center = (point.Point2D(x1,y1)) #координата скважины

class candidat():
    def __init__(self, p, i):
        self.point = p
        self.intersec = i