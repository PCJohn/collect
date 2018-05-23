import numpy as np
import tensorflow as tf

import mnist

def start_sess():
    config = tf.ConfigProto()
    sess = tf.Session(config=config)
    return sess

def acc(py,y):
    return np.mean(py.argmax(axis=1)==y.argmax(axis=1))

class SMax():
    def __init__(self):
        self.lr = 1e-4
        self.bz = 100
        self.eps = 1e-8
        self.epch = 1000

        self.x = tf.placeholder(tf.float32,shape=[None,3,28,28])
        self.y = tf.placeholder(tf.float32,shape=[None,n_classes])
        self.keep_prob = tf.placeholder(tf.float32)

        out = self.x
        out = tf.layers.conv2d(inputs=out,filters=32,kernel_size=[3,3],padding='same',activation=tf.nn.relu,strides=2)
        #out = tf.layers.max_pooling2d(inputs=out,pool_size=[2,2],padding='same',strides=2)
        out = tf.layers.conv2d(inputs=out,filters=48,kernel_size=[3,3],padding='same',activation=tf.nn.relu,strides=2)
        #out = tf.layers.max_pooling2d(inputs=out,pool_size=[2,2],padding='same',strides=2)
        out = tf.reshape(out,[-1,np.prod(out.get_shape().as_list()[1:])])
        
        #out = tf.layers.dense(inputs=out,units=800,activation=tf.nn.relu)
        #out = tf.layers.dropout(inputs=out,rate=self.keep_prob)
        
        #out = tf.layers.dense(inputs=out,units=400,activation=tf.nn.relu)
        #out = tf.layers.dropout(inputs=out,rate=self.keep_prob)
        
        #SMax concat part
        #smz = ([100]*2)
        #sm = []
        #for sm_size in smz:
        #    sm.append(tf.layers.dense(inputs=out,units=sm_size,activation=tf.nn.softmax))
        #out = tf.concat(sm,1)
        ################

        #SMax concat part
        smz = ([200]*1)
        sm = []
        for sm_size in smz:
            sml = tf.layers.dense(inputs=out,units=sm_size,activation=tf.nn.relu)
            #sml = tf.nn.softmax(sml)
            sm.append(sml)
        feat = tf.concat(sm,1)
        self.feat = feat
        ################
        
        #Directly use a really low dimension embedding 
        #out = tf.layers.dense(inputs=out,units=200,activation=tf.nn.relu)
        #feat = tf.layers.dropout(inputs=out,rate=self.keep_prob)
        #self.feat = feat
        ################
        
        self.feat_dim = feat.get_shape().as_list()[-1]
        self.out = tf.layers.dense(inputs=feat,units=n_classes,trainable=True)

        self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.y,logits=self.out))
        self.train_step = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.loss)
        #self.train_step = tf.train.GradientDescentOptimizer(learning_rate=self.lr).minimize(self.loss)
        self.sess = None

    def preproc(self,x):
        return x

    def train(self,ds,vds):
        x,y = map(np.array,zip(*ds))
        vx,vy = map(np.array,zip(*vds))
        print 'Dataset shape:',x.shape,'--',y.shape,'__',vx.shape,'--',vy.shape
        
        x = self.preproc(x)
        vx = self.preproc(vx)
        
        ind = range(x.shape[0])
        if (self.sess is None) or (self.sess._closed):
            self.sess = start_sess()
            self.sess.run(tf.global_variables_initializer())
        for epch in range(self.epch):
            bi = np.random.choice(ind,self.bz)
            bx,by = x[bi],y[bi]
            feed_dict = {self.x:bx,self.y:by,self.keep_prob:0.5}
            _,loss = self.sess.run([self.train_step,self.loss],feed_dict=feed_dict)
            vpy = self.sess.run(self.out,feed_dict={self.x:vx,self.y:vy,self.keep_prob:1})
            vacc = acc(vpy,vy)
            if epch%100 == 0:
                print 'Batch',epch,'\tLoss:',loss,'\tAcc',vacc

    def extract_feat(self,x):
        return self.sess.run(self.feat,feed_dict={self.x:x,self.keep_prob:1})

    def save(self,path):
        saver = tf.train.Saver()
        saver.save(self.sess,path)

    def load(self,path):
        if (self.sess is None) or (self.sess._closed):
            self.sess = start_sess()
            self.sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        saver.restore(self.sess,path)

    def close(self):
        self.sess.close()

n_classes = 5
"""ds,vds = mnist.load(n_classes=n_classes)
model = SMax()
model.train(ds,vds)
path = './smaxmodel'
model.save(path)
model.close()"""
#model.load(path)
#model.train(ds,vds)
#model.save(path)
