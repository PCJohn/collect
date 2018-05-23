import os
import cv2
import random
import numpy as np

path = '/home/prithvi/dsets/MNIST/trainingSet/'
train_count = 3000
val_count = 1000

channel_first = True
col = False

def noisy(noise_typ,image):
   if noise_typ == "gauss":
      row,col,ch= image.shape
      mean = 0
      var = 0.1
      sigma = var**0.5
      gauss = np.random.normal(mean,sigma,(row,col,ch))
      gauss = gauss.reshape(row,col,ch)
      noisy = image + gauss
      return noisy
   elif noise_typ == "s&p":
      row,col,ch = image.shape
      s_vs_p = 0.5
      amount = 0.004
      out = np.copy(image)
      # Salt mode
      num_salt = np.ceil(amount * image.size * s_vs_p)
      coords = [np.random.randint(0, i - 1, int(num_salt))
              for i in image.shape]
      out[coords] = 1
      # Pepper mode
      num_pepper = np.ceil(amount* image.size * (1. - s_vs_p))
      coords = [np.random.randint(0, i - 1, int(num_pepper))
              for i in image.shape]
      out[coords] = 0
      return out
   elif noise_typ == "poisson":
      vals = len(np.unique(image))
      vals = 2 ** np.ceil(np.log2(vals))
      noisy = np.random.poisson(image * vals) / float(vals)
      return noisy
   elif noise_typ =="speckle":
      row,col,ch = image.shape
      gauss = np.random.randn(row,col,ch)
      gauss = gauss.reshape(row,col,ch)        
      noisy = image + image * gauss
      return noisy

def load(shuffle=True,size=(28,28),train_count=3000,val_count=1000,channel_first=True,col=True,flatten=False,norm=True,add_noise=True,n_classes=10):
    ds = []
    vds = []
    classes = os.listdir(path)
    classes.sort()
    #unit = np.diag(np.ones(len(classes)))
    unit = np.diag(np.ones(n_classes))
    #blank = np.zeros()
    for n in classes[:n_classes]:
        n_path = os.path.join(path,n)
        lab = unit[int(n)]
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
            if add_noise == True:
                nimg = noisy('gauss',img)
                print img.shape
                cv2.imshow('nimg',nimg.transpose(2,1,0))
                cv2.waitKey(0)
                cv2.destroyAllWindows()
                ds.append((nimg,lab))
        #print len(vds),
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
    if shuffle == True:
        random.shuffle(ds)
        random.shuffle(vds)
    #X,Y = zip(*ds)
    return (ds,vds)

ds,vds = load()
"""for x,y in ds[:10]:
    print x.shape
    cv2.imshow(str(y),x)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
"""
