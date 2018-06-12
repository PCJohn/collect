import os
import cv2
import random
import numpy as np

path = '../gan/unsup_data'
train_count = 3000
val_count = 1000

channel_first = True
col = True

def load(shuffle=True,size=(256,256),train_count=3000,val_count=975,channel_first=True,col=True,flatten=False,norm=True,n_classes=10):
    ds = []
    vds = []
    classes = os.listdir(path)
    #np.random.shuffle(classes)
    classes.sort()
    #unit = np.diag(np.ones(len(classes)))
    unit = np.diag(np.ones(n_classes))
    #blank = np.zeros()
    for k,n in enumerate(classes[:n_classes]):
        n_path = os.path.join(path,n)
        #lab = unit[int(n)]
        lab = unit[k]
        #lab = [0.1*int(n)]
        flist = os.listdir(n_path)
        random.shuffle(flist)
        for s in flist[:train_count]:
            if col == True:
                img = cv2.imread(os.path.join(n_path,s))
            else:
                img = cv2.imread(os.path.join(n_path,s),0)
                img = img[...,np.newaxis]
            img = cv2.resize(img,size)
            if channel_first == True:
                img = img.transpose(2,1,0)
            if norm == True:
                img = np.float32(img)/255.
            if flatten == True:
                img = img.flatten()
            ds.append((img,lab))
        
        for s in flist[train_count:train_count+val_count]:
            if col == True:
                img = cv2.imread(os.path.join(n_path,s))
            else:
                img = cv2.imread(os.path.join(n_path,s),0)
                img = img[...,np.newaxis]
            img = cv2.resize(img,size)
            if channel_first == True:
                img = img.transpose(2,1,0)
            if norm == True:
                img = np.float32(img)/255.
            if flatten == True:
                img = img.flatten()
            vds.append((img,lab))
        #print len(vds),'<=='
    if shuffle:
        random.shuffle(ds)
        random.shuffle(vds)
    return (ds,vds)

ds,vds = load()
for x,y in ds[:10]:
    print(x.shape)
    cv2.imshow(str(y),x.transpose(2,1,0))
    cv2.waitKey(0)
    cv2.destroyAllWindows()
x_train,y_train = map(np.array,zip(*ds))
print(x_train.shape,y_train.shape)
