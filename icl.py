import numpy as np
import tensorflow as tf

import mnist
from smax import SMax
from cons import Cons

def start_sess():
    config = tf.ConfigProto()
    sess = tf.Session(config=config)
    return sess

def acc(py,y):
    return np.mean(py.argmax(axis=1)==y.argmax(axis=1))

class Icl():
    def __init__(self,feature_model):
        self.lr = 1e-4
        self.bz = 30
        self.eps = 1e-8
        self.epch = 5000

        self.featm = feature_model
        self.x = tf.placeholder(tf.float32,shape=[None,self.featm.feat_dim])
        self.y = tf.placeholder(tf.float32,shape=[None,n_classes])
        self.keep_prob = tf.placeholder(tf.float32)

        out = self.x
        out = tf.layers.dense(inputs=out,units=200,activation=tf.nn.relu)
        out = tf.layers.dropout(inputs=out,rate=self.keep_prob)
        self.out = tf.layers.dense(inputs=out,units=n_classes)

        self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.y,logits=self.out))
        self.train_step = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.loss)
        self.sess = None

    def train(self,ds,vds):
        x,y = map(np.array,zip(*ds))
        vx,vy = map(np.array,zip(*vds))
        print 'Dataset shape:',x.shape,'--',y.shape,'__',vx.shape,'--',vy.shape
        
        x = self.featm.preproc(x)
        vx = self.featm.preproc(vx)
       
        fx = self.featm.extract_feat(x)
        print np.round(fx[0]),fx[0].max()
        vfx = self.featm.extract_feat(vx)
        
        ind = range(x.shape[0])
        self.sess = start_sess()
        self.sess.run(tf.global_variables_initializer())
        for epch in range(self.epch):
            bi = np.random.choice(ind,self.bz)
            bx,by = fx[bi],y[bi]
            feed_dict = {self.x:bx,self.y:by,self.keep_prob:0.5}
            _,loss = self.sess.run([self.train_step,self.loss],feed_dict=feed_dict)
            vpy = self.sess.run(self.out,feed_dict={self.x:vfx,self.y:vy,self.keep_prob:1})
            vacc = acc(vpy,vy)
            if epch%100 == 0:
                print 'Batch',epch,'\tLoss:',loss,'\tAcc',vacc

    def predict(self):
        pass

#ds,vds = mnist.load(n_classes=5)
smpath = './smaxmodel'
smf = SMax()
#smf.train(ds,vds)
smf.load(smpath)

"""conspath = './consmodel'
cmf = Cons()
cmf.load(conspath)"""

n_classes = 10
tc = 3
vc = 1000
ds,vds = mnist.load(n_classes=n_classes,train_count=tc,val_count=vc)
model = Icl(smf)
model.train(ds,vds)
