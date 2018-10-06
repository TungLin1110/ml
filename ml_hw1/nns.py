import numpy as np
import matplotlib.pyplot as plt
import os
from skimage import io
from PIL import Image
import sys
#for i in range(39):
dir = (os.listdir('CroppedYale'))  # find path
dir.sort()
#print(len(dir))
#for i in range(39):
#    print(dir[i+1])
num = 0

train_set = {}  # store training set 38 label
test_set = []  # store testing set
test_label = []
for i in range(len(dir)-1):
    path = 'CroppedYale/' + dir[i+1]
    name_str = dir[i+1].strip('yaleB')
    name_int = int(name_str)
    dir2 = os.listdir(path)
    dir2.sort()
    train = []
    for j in range(64):
        if j < 35:
            train.append(np.array(Image.open(path+'/'+dir2[j]), dtype=float))
            num += 1
        else:
            test_set.append(
                np.array(Image.open(path+'/'+dir2[j]), dtype=float))
            test_label.append(name_int)
            num += 1
    train_set[name_int] = train

print (num)
'''
for i in range(35):
    print(test_label[i],test_set[i])
'''

def SSD(im1, im2):
    #x = np.array(im1)
    #y = np.array(im2)
    return np.sum((im1 - im2)**2)


def SAD(im1, im2):
    #x = np.array(im1)
    #y = np.array(im2)
    return np.sum(abs(im1 - im2))


#print(dir[1])
#im=SSD(ima,imb)
#print(im)

ssdnum = 0
sadnum = 0
totalnum = 0
for i in range(len(test_set)):
    SSD_hit = False
    SAD_hit = False
    ssdmin = sys.maxsize
    sadmin = sys.maxsize

    for j in range(1, len(train_set)+2):
        if(j == 14):
            continue
        for k in range(len(train_set[j])):
            #ima = Image.open(test_set[i])
            #imb = Image.open(train_set[j][k])
            SSD_local = np.sum((test_set[i] - train_set[j][k])**2)
            #if(SSD(test_set[i], train_set[j][k]) < ssdmin):
            if(SSD_local<ssdmin):
            #    ssdmin = SSD(test_set[i], train_set[j][k])
                ssdmin = SSD_local
                #checka = test_set[i]
                #checkb = train_set[j][k]
                #ssdlabela = test_label[i]
                #ssdlabelb = j
                if(test_label[i] == j):
                    SSD_hit = True
                else:
                    SSD_hit = False
            SAD_local = np.sum(np.absolute(test_set[i]-train_set[j][k]))
            #if(SAD(test_set[i], train_set[j][k]) < sadmin):
            if(SAD_local < sadmin):
            #    sadmin = SAD(test_set[i], train_set[j][k])
                sadmin = SAD_local
                #sadlabela = test_label[i]
                #sadlabelb = j
                if(test_label[i] == j):
                    SAD_hit = True
                else:
                    SAD_hit = False
        #print(test_set[i],train_set[j][k])
    totalnum += 1
    #if(ssdlabela == ssdlabelb):
    #    ssdnum += 1
    #if(sadlabela == sadlabelb):
    #    sadnum += 1
    if SSD_hit:
        ssdnum += 1
    if SAD_hit:
        sadnum += 1
    print(f'total:{totalnum},ssd:{ssdnum},sad:{sadnum}')
    #print(checka,checkb)
print(f'ssd:{ssdnum/totalnum}')
print(f'sad:{sadnum/totalnum}')
