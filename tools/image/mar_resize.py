import numpy as np
from PIL import Image
import cv2
def pad_resize(image,target_size,cval=0): #target_size=(target_w,target_h)
    target_w=target_size[0]
    target_h=target_size[1]
    returnPIL=False
    if isinstance(image,Image.Image):
        image = np.asarray(image)
        returnPIL=True
    elif isinstance(image,np.ndarray):
        pass
    else:
        raise Exception("Unsupport format!")
    image_h = image.shape[0]
    image_w = image.shape[1]
    rate_w=target_w /image_w
    rate_h=target_h /image_h
    if rate_w<rate_h:       #取最小放大比例
        no_padding_result=cv2.resize(image,(target_w,int(image_h*rate_w)))        #宽放大比例最小，填充宽
    else:
        no_padding_result=cv2.resize(image, (int(image_w*rate_h), target_h))  # 高放大比例最小，填充高
    no_padding_h = no_padding_result.shape[0]
    no_padding_w = no_padding_result.shape[1]
    pad_h=target_h-no_padding_h
    pad_w=target_w-no_padding_w
    if len(no_padding_result.shape)!=2:
        result=np.lib.pad(no_padding_result, ((int(pad_h / 2), pad_h - int(pad_h / 2)),
                                (int(pad_w / 2), pad_w - int(pad_w / 2)), (0, 0)),
                        'constant', constant_values=cval)
    else:
        result = np.lib.pad(no_padding_result, ((int(pad_h / 2), pad_h - int(pad_h / 2)),
                                                (int(pad_w / 2), pad_w - int(pad_w / 2))),
                            'constant', constant_values=cval)
    if returnPIL:
        return Image.fromarray(np.uint8(result))
    else:
        return result


def cut_resize(image, target_size):
    target_w = target_size[0]
    target_h = target_size[1]
    returnPIL = False
    if isinstance(image, Image.Image):
        image = np.asarray(image)
        returnPIL = True
    elif isinstance(image, np.ndarray):
        pass
    else:
        raise Exception("Unsupport format!")
    image_h = image.shape[0]
    image_w = image.shape[1]
    rate_w = target_w / image_w
    rate_h = target_h / image_h
    if rate_w > rate_h:  # 取最大放大比例
        cut_h=image_h-int(target_h/rate_w)
        if (cut_h==0):
            cut_result=image
        else:
            if len(image.shape) != 2:
                cut_result =image[int(cut_h/2):-(cut_h-int(cut_h/2)) ,::]   # 宽放大比例最大，去除高
            else:
                cut_result = image[int(cut_h / 2):-(cut_h - int(cut_h / 2))]
    else:
        cut_w=image_w-int(target_w/rate_h)
        if (cut_w==0):
            cut_result=image
        else:
            if len(image.shape) != 2:
                cut_result = image[:,int(cut_w / 2):-(cut_w - int(cut_w / 2)),::]  # 高放大比例最小，去除宽
            else:
                cut_result = image[:, int(cut_w / 2):-(cut_w - int(cut_w / 2))]

    result = cv2.resize(cut_result, (target_size[0], target_size[1]))
    if returnPIL:
        return Image.fromarray(np.uint8(result))
    else:
        return result

if __name__ == '__main__':

##Pad_resize Test
    testimg=Image.open("test0.bmp")
    print("input size:",testimg.size)
    resultimg=pad_resize(testimg,(600,300),0)
    print("output size:", resultimg.size)
    resultimg.save("test0_rgb_pad.bmp")

    testimg=Image.open("test0.bmp").convert('L')
    print("input size:",testimg.size)
    resultimg=pad_resize(testimg,(600,300),255)
    print("output size:", resultimg.size)
    resultimg.save("test0_L_pad.bmp")


    testimg=Image.open("test1.bmp")
    print("input size:",testimg.size)
    resultimg=pad_resize(testimg,(300,600),127)
    print("output size:", resultimg.size)
    resultimg.save("test1_rgb_pad.bmp")

    testimg=Image.open("test1.bmp").convert('L')
    print("input size:",testimg.size)
    resultimg=pad_resize(testimg,(300,600),0)
    print("output size:", resultimg.size)
    resultimg.save("test1_L_pad.bmp")



##cut_resize Test
    testimg=Image.open("test0_rgb_pad.bmp")
    print("input size:",testimg.size)
    resultimg=cut_resize(testimg,(400,300))
    print("output size:", resultimg.size)
    resultimg.save("test0_rgb_cut_pad.bmp")

    testimg=Image.open("test0_L_pad.bmp").convert('L')
    print("input size:",testimg.size)
    resultimg=cut_resize(testimg,(400,300))
    print("output size:", resultimg.size)
    resultimg.save("test0_L_cut_pad.bmp")

    testimg=Image.open("test1_rgb_pad.bmp")
    print("input size:",testimg.size)
    resultimg=cut_resize(testimg,(300,400))
    print("output size:", resultimg.size)
    resultimg.save("test1_rgb_cut_pad.bmp")


    testimg=Image.open("test1_L_pad.bmp").convert('L')
    print("input size:",testimg.size)
    resultimg=cut_resize(testimg,(300,400))
    print("output size:", resultimg.size)
    resultimg.save("test1_L_cut_pad.bmp")






