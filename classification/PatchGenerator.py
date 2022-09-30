from segmentation.dataset.Kvasir2Keras import DATA_PATH,IMG_FMT,MASK_FMT
import os.path as osp
import os, shutil
import cv2
from glob import glob
import random
import numpy as np
from tools.image.border import GetEdgeInfo



def get_ran(ranminsize,shape):
    flag=True
    while flag:
        y=random.randint(0,shape[0]-ranminsize)
        x=random.randint(0,shape[1]-ranminsize)
        h=random.randint(ranminsize,shape[0]-y)
        wmin=int(max(ranminsize,h*0.5))
        wmax=int(min(shape[1]-x,h*2))
        if (wmin<wmax):
            w=random.randint(wmin,wmax)
            return x, y, w, h

def judge_not_cross_rect(rect1,rect2):
    minx1,miny1,maxx1,maxy1=rect1[0],rect1[1],rect1[0]+rect1[2],rect1[1]+rect1[3]
    minx2,miny2,maxx2,maxy2=rect2[0],rect2[1],rect2[0]+rect2[2],rect2[1]+rect2[3]
    minx = max(minx1, minx2)
    miny = max(miny1, miny2)
    maxx = min(maxx1, maxx2)
    maxy = min(maxy1, maxy2)
    if minx > maxx or miny > maxy:
        return True
    else:
        return False


def gen_patch(img,seg,filename,npcount=3,npsize=50):
    preview1=img.copy()
    ret, thresh = cv2.threshold(seg, 0, 255, 0)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, 2)
    count=0
    bbpolyp=[]
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        # print(x,y,w,h)
        if w*h<=6:
            print("Warning: Find very small polyp patch from", filename,",skip it.")
        else:
            bbpolyp.append(cv2.boundingRect(cnt))
            cv2.rectangle(preview1, (x, y), (x + w, y + h), (0, 255, 0), 2)
            count = count + 1
    if(count>1):
        print("Warning: Find",count,"polyp patches from", filename)
    preview = preview1.copy()
    count = 0
    bbnone = []
    skipcount=0
    while count<npcount:
        ran=get_ran(npsize,preview.shape)
        judge=True
        for bb in bbpolyp:
            if(not judge_not_cross_rect(bb,ran)):
                judge=False
        for bb in bbnone:
            if (not judge_not_cross_rect(bb, ran)):
                judge = False
        if judge:
            bbnone.append(ran)
            count=count+1
            cv2.rectangle(preview, (ran[0], ran[1]), (ran[0] + ran[2], ran[1] + ran[3]), (255, 0, 0), 2)
            skipcount=0
        else:
            skipcount+=1
            if skipcount>1000:
                npsize = int(npsize * .98)
                if npsize<20:
                    print("Warning: Can not find any more non-polyp patch from", filename,"Found:",count,"non-polyp patches")
                    break
                else:
                    print("Warning: Can not find any more non-polyp patch from",filename,"change min size to",npsize,"and retry.")
                    count=0
                    skipcount=0
                    bbnone = []
                    preview = preview1.copy()

    return bbpolyp,bbnone,preview

def convert():
    for subset in ['train','val','test']:
        os.makedirs(osp.join(DATA_PATH,'Classification','Preview',subset))
        os.makedirs(osp.join(DATA_PATH,'Classification',subset,'Polyp'))
        os.makedirs(osp.join(DATA_PATH,'Classification',subset,'NonPolyp'))
        all_images=glob(osp.join(DATA_PATH,'Segmentation',subset,'image','0', '*'+IMG_FMT))
        images=[osp.basename(image) for image in all_images]
        for image in images:
            img=cv2.imread(osp.join(DATA_PATH,'Segmentation',subset,'image','0',image))
            seg=cv2.imread(osp.join(DATA_PATH,'Segmentation',subset,'mask','0',osp.splitext(image)[0]+MASK_FMT), cv2.IMREAD_GRAYSCALE)
            wbegin, hbegin, width, height=GetEdgeInfo(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            img=img[hbegin:hbegin + height, wbegin:wbegin + width]
            seg=seg[hbegin:hbegin + height, wbegin:wbegin + width]
            bbpolyp, bbnone, preview=gen_patch(img,seg,image,3,npsize=int(img.shape[1]*0.3))
            cv2.imwrite(os.path.join(DATA_PATH,'Classification','Preview',subset,  osp.splitext(image)[0]  + IMG_FMT), preview)
            i=0
            for bb in bbpolyp:
                x,y,w,h=bb
                cv2.imwrite(os.path.join(DATA_PATH,'Classification', subset,'Polyp',  osp.splitext(image)[0] +'_'+str(i) + IMG_FMT), img[y:y+h, x:x+w])
                i+=1
            i = 0
            for bb in bbnone:
                x,y,w,h=bb
                cv2.imwrite(os.path.join(DATA_PATH,'Classification',subset, 'NonPolyp',  osp.splitext(image)[0] +'_'+str(i) + IMG_FMT), img[y:y+h, x:x+w])
                i += 1

if __name__ == '__main__':
    if osp.exists(osp.join(DATA_PATH, 'Classification')):
        shutil.rmtree(osp.join(DATA_PATH, 'Classification'))
    convert()


