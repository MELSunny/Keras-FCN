
import numpy as np
from skimage import color
from skimage.filters.rank import median
from skimage import img_as_ubyte

def CountSame(array):#Count same value at begin and end
    begin=0
    for num in array:
        if num<26:
            begin=begin+1
        else:
            break

    end=0
    for num in array[::-1]:
        if num<26:
            end=end+1
        else:
            break
    return begin,end

def GetEdgeInfo(image):
    grey = color.rgb2gray(image)
    grey=img_as_ubyte(grey)
    h_img,w_img=grey.shape
    med = median(grey)
    w_max = np.amax(med, axis=0)
    h_max = np.amax(med, axis=1)
    wbegin,wend=CountSame(w_max)
    hbegin,hend=CountSame(h_max)
    height=h_img-hend-hbegin
    width=w_img-wend-wbegin
    return [wbegin,hbegin,width,height]

def CutEdge(image):
    [wbegin, hbegin, width, height]=GetEdgeInfo(image)
    return image[hbegin:hbegin + height, wbegin:wbegin + width]