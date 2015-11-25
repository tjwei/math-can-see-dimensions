from IPython.display import HTML, display, YouTubeVideo, VimeoVideo, Image
import numpy as np
import matplotlib as mpl
from matplotlib import figure
from matplotlib.pyplot import *
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from numpy import arange, sqrt, cos, pi, dot

np.set_printoptions(suppress=True, precision=3)
def show_image(img, interpolation=None):
    figure(figsize=(10,6))
    plt.imshow(img, interpolation=interpolation)
    plt.colorbar()
def chopped(original, level=5, level2=0, N=8):
    chopped = original.copy()
    for x in range(N):
        for y in range(N):
            if x+y > level or x+y<level2:
                chopped[x,y] = 0.
    return chopped
def dct_all(img, level): 
    img2 = img.copy()
    for i in range(0,img.shape[0]-8,8):
        for j in range(0,img.shape[1]-8,8):
            tinyDCT = doDCT(img2[i:i+8,j:j+8])
            tinyDCT_chopped = chopped(tinyDCT, level)
            img2[i:i+8,j:j+8] = undoDCT(tinyDCT_chopped)
    return img2
def generate_dct(N=8, step=1):
    dct = np.zeros((N, len(arange(0,8,step))))
    dct[0] = sqrt(2/N) / sqrt(2.0)
    for u in range(1,N):
        x = arange(0, 8, step)
        dct[u] = sqrt(2/N) * cos( pi/N * u*(x+0.5) )
    return dct
dct = generate_dct()
dct0= generate_dct(step=0.1)
def show_dct_fig():
    figure(figsize=(12,7))
    for u in range(8):
        subplot(2, 4, u+1)
        ylim((-1, 1))
        title(str(u))
        plot(arange(0,8,0.1), dct0[u, :])
        plot(dct[u, :],'ro')
def doDCT(f):
    return dot(dot(dct, f), dct.T)

def undoDCT(G):
    return dot(dot(dct.T, G), dct)
# a wrapper of word2vec api of model
import word2vec
model = word2vec.load('sdyxz.bin')
class w:
    def __init__(self, word=None):
        self.pos = [word] if word else []
        self.neg = []
    def __add__(self, other):
        rtn = w()
        rtn.pos = self.pos+other.pos
        rtn.neg = self.neg+other.neg
        return rtn
    def __sub__(self, other):
        rtn = w()
        rtn.pos = self.pos+other.neg
        rtn.neg = self.neg+other.pos
        return rtn
    def __neg__(self):
        rtn = w()
        rtn.pos = self.neg
        rtn.neg = self.pos
        return rtn
    def analogy(self, n=6):
        indexes, metrics = model.analogy(pos=self.pos, neg=self.neg, n=6)
        return model.generate_response(indexes, metrics).tolist()
    def __repr__(self):
        return "\n".join(map(repr, self.analogy()))
