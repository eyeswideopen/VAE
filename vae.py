#All Imports
import sys
!{sys.executable} -m pip install imageio
import numpy as np
import imageio
import os.path
import scipy
import tensorflow as tf
import matplotlib.pyplot as pyplot


#os.environ["CUDA_VISIBLE_DEVICES"]="4"

from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())

orderedFiles = "/data/master_thesis_koerner/master_thesis/data/bones/raw/labels"

#Bilder_liste = [f for f in sorted(os.listdir(orderedFiles)) if isfile(join(orderedFiles, f)) and "image" in f]
segmentierung_liste = [f for f in sorted(os.listdir(orderedFiles)) if os.path.isfile(os.path.join(orderedFiles, f)) and ".png" in f]
#print(len(segmentierung_liste))

labels = np.random.randint(5, size=len(segmentierung_liste))


image_size = 256



def next_batch(num, data, labels):
    '''
    Return a total of `num` random samples and labels. 
    '''
    idx = np.arange(0 , len(data))
    np.random.shuffle(idx)
    idx = idx[:num]
    data = []
    label = []
    for i in idx:
        #label.append(labels[i])
        data.append(scipy.misc.imresize(imageio.imread(os.path.join(orderedFiles, segmentierung_liste[i])),(image_size, image_size), interp='nearest'))
        label.append(scipy.misc.imresize(imageio.imread(os.path.join(orderedFiles, segmentierung_liste[i])),(image_size, image_size), interp='nearest'))           
    data = np.asarray(data)
    label = np.asarray(label)
    #data_shuffle = [data[ imageio.imread(os.path.join(orderedFiles, segmentierung_liste[i]))] for i in idx]
    #labels_shuffle = [labels[ i] for i in idx]
    #arr=[]
    #for t in zip(data,label):
    #    vals = {}
    #    vals['X_in']=t[0]
    #    vals['Y']=t[1]
    #    arr.append(vals)  
    return data, label


tf.reset_default_graph()

batch_size = 32
image_size = 256
X_in = tf.placeholder(dtype=tf.float32, shape=[None, image_size, image_size], name='X')
Y    = tf.placeholder(dtype=tf.float32, shape=[None, image_size, image_size], name='Y')
Y_flat = tf.reshape(Y, shape=[-1, image_size * image_size])
keep_prob = tf.placeholder(dtype=tf.float32, shape=(), name='keep_prob')

dec_in_channels = 1
n_latent = 8
#3x3->7x7
#49->9

reshaped_dim = [-1, 7, 7, dec_in_channels]
inputs_decoder = 49 * dec_in_channels / 2


def lrelu(x, alpha=0.3):
    return tf.maximum(x, tf.multiply(x, alpha))

def encoder(X_in, keep_prob):
    activation = lrelu
    with tf.variable_scope("encoder", reuse=None):
            X = tf.reshape(X_in, shape=[-1, image_size, image_size, 1])
            x = tf.layers.conv2d(X, filters=64, kernel_size=4, strides=2, padding='same', activation=activation)
            x = tf.nn.dropout(x, keep_prob)
            x = tf.layers.conv2d(x, filters=64, kernel_size=4, strides=2, padding='same', activation=activation)
            x = tf.nn.dropout(x, keep_prob)
            x = tf.layers.conv2d(x, filters=64, kernel_size=4, strides=1, padding='same', activation=activation)
            x = tf.nn.dropout(x, keep_prob)
            x = tf.contrib.layers.flatten(x)
            mn = tf.layers.dense(x, units=n_latent)
            sd       = 0.5 * tf.layers.dense(x, units=n_latent)            
            epsilon = tf.random_normal(tf.stack([tf.shape(x)[0], n_latent])) 
            z  = mn + tf.multiply(epsilon, tf.exp(sd))
            return z, mn, sd

def decoder(sampled_z, keep_prob):
    with tf.variable_scope("decoder", reuse=None):
            x = tf.layers.dense(sampled_z, units=int(inputs_decoder), activation=lrelu)
            x = tf.layers.dense(x, units=int(inputs_decoder) * 2 + 1, activation=lrelu)
            x = tf.reshape(x, reshaped_dim)
            x = tf.layers.conv2d_transpose(x, filters=64, kernel_size=4, strides=1, padding='same', activation=tf.nn.relu)
            x = tf.nn.dropout(x, keep_prob)
            x = tf.layers.conv2d_transpose(x, filters=64, kernel_size=4, strides=1, padding='same', activation=tf.nn.relu)
            x = tf.nn.dropout(x, keep_prob)
            x = tf.layers.conv2d_transpose(x, filters=64, kernel_size=4, strides=2, padding='same', activation=tf.nn.relu)
            x = tf.contrib.layers.flatten(x)
            x = tf.layers.dense(x, units=image_size*image_size, activation=tf.nn.sigmoid)
            img = tf.reshape(x, shape=[-1, image_size, image_size])
            return img


sampled, mn, sd = encoder(X_in, keep_prob)
dec = decoder(sampled, keep_prob)


unreshaped = tf.reshape(dec, [-1, image_size*image_size])
img_loss = tf.reduce_sum(tf.squared_difference(unreshaped, Y_flat), 1)
latent_loss = -0.5 * tf.reduce_sum(1.0 + 2.0 * sd - tf.square(mn) - tf.exp(2.0 * sd), 1)
loss = tf.reduce_mean(img_loss + latent_loss)
optimizer = tf.train.AdamOptimizer(0.00030).minimize(loss)
sess = tf.Session()
sess.run(tf.global_variables_initializer())


for i in range(20000):
    #batch = [np.reshape(b, [image_size, image_size]) for b in mnist.train.next_batch(batch_size=batch_size)[0]]
    #batch = [np.reshape(b, [image_size, image_size]) for b in next_batch(batch_size, segmentierung_liste, labels)[0]]
    data = next_batch(batch_size, segmentierung_liste, labels)[0]
    label = next_batch(batch_size, segmentierung_liste, labels)[1]
    data = data.reshape([batch_size, image_size, image_size])
    sess.run(optimizer, feed_dict = {X_in: data, Y: data, keep_prob: 0.8})
    
    if not i % 25:
        ls, d, i_ls, d_ls, mu, sigm = sess.run([loss, dec, img_loss, latent_loss, mn, sd], feed_dict = {X_in: data, Y: data, keep_prob: 1.0})
        plt.imshow(np.reshape(data[0], [image_size, image_size]), cmap='gray')
        plt.show()
        plt.imshow(d[0], cmap='gray')
        plt.show()
        print(i, ls, np.mean(i_ls), np.mean(d_ls))


#Store Model
#Create a saver object which will save all the variables
saver = tf.train.Saver()
saver.save(sess, '../vae/vea50000',global_step=50000)



#create new labels
import cv2
orderedFiless = "/data/master_thesis_koerner/master_thesis/data/bones/vae/labels"
for j in range(500):
    randoms = [np.random.normal(0, 1, n_latent) for _ in range(100)]

    imgs = sess.run(dec, feed_dict = {sampled: randoms, keep_prob: 1.0})
    imgs = [np.reshape(imgs[i], [image_size, image_size]) for i in range(len(imgs))]
    i = 0
    for img in imgs[:9999]:
        i = i+1
        plt.figure(figsize=(5,5))
        plt.figure(figsize=(5,5))
        plt.axis('off')
        img = cv2.medianBlur(img, 3)
        high_values = img < 0.2  # Where values are low
        img[high_values] = 0
        img = cv2.medianBlur(img, 3)
        fac= 0.9
        img[img < fac] = 0
        img[img >=(1-fac)] = 1
        #img = scipy.misc.imresize(img,(512, 512), interp='nearest')
        img = cv2.medianBlur(img, 3)
        #print(i)
        plt.imshow(img, cmap='gray')
        #if np.count_nonzero(img == 0) <=4:
        #    continue
        #print(np.unique(img))
        if 255 not in np.unique(img):
            continue
        #plt.imshow(img, cmap='gray')
        print(i)
        imageio.imwrite(os.path.join(orderedFiless, "vae"+str(j)+str(i)+".png"),img, "png")

    