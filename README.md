## Variaional AutoEncoder-Tensorflow

**I Write the code of Variational AutoEncoder using the tensorflow**

**This code has following features**
1. when we train our model, I use 0.6 dropout rate.
2. All activation functions are elu.
3. I use Xavier_initializer for weights initialization.


## Enviroment
**- OS: window 10(64bit)**

**- Python 3.5**

**- Tensorflow-gpu version:  1.3.0rc2**


## Schematic of VAE

![사진1](https://github.com/MINGUKKANG/VAE_tensorflow/blob/master/image/model.PNG)

## Code

**1. Gaussian Encoder**

```python
def Bernoulli_decoder(z, n_hidden, n_out ,keep_prob):
    w_init = tf.contrib.layers.xavier_initializer()
    z_shape = z.get_shape()
    
    with tf.variable_scope("decoder_hidden_1", reuse = tf.AUTO_REUSE):
        w4 = tf.get_variable("w4", shape = [z_shape[1],n_hidden], initializer = w_init)
        b4 = tf.get_variable("b4", shape = [n_hidden], initializer = tf.constant_initializer(0.))
        h4 = tf.matmul(z,w4) + b4
        h4 = tf.nn.elu(h4)
        h4 = tf.nn.dropout(h4,keep_prob)
        
    with tf.variable_scope("decoder_hidden_2", reuse = tf.AUTO_REUSE):
        w5 = tf.get_variable("w5", shape = [n_hidden, n_hidden], initializer = w_init)
        b5 = tf.get_variable("b5", shape = [n_hidden], initializer = tf.constant_initializer(0.))
        h5 = tf.matmul(h4,w5) + b5
        h5 = tf.nn.elu(h5)
        h5 = tf.nn.dropout(h5, keep_prob)
        
    with tf.variable_scope("decoder_output", reuse = tf.AUTO_REUSE):
        w6 = tf.get_variable("w6",shape = [n_hidden, n_out], initializer = w_init)
        b6 = tf.get_variable("b6", shape = [n_out], initializer = tf.constant_initializer(0.))
        h6 = tf.matmul(h5,w6) + b6
        h6 = tf.nn.sigmoid(h6)
        
        return h6
```

**2. Bernoulli Decoder**

![사진3](https://github.com/MINGUKKANG/VAE_tensorflow/blob/master/image/Bernoulli_decorder.PNG)

**3. Variational AutoEncoder**

![사진4](https://github.com/MINGUKKANG/VAE_tensorflow/blob/master/image/Variational_Autoencoder.PNG)

## Result
**1. Comparing the generated images with the original images**

![사진5](https://github.com/MINGUKKANG/VAE_tensorflow/blob/master/image/Result1.PNG)

**2. Distribution of MNIST Data**

![사진6](https://github.com/MINGUKKANG/VAE_tensorflow/blob/master/image/result2.png)

**3. Manifold of MNIST Data**

![사진7](https://github.com/MINGUKKANG/VAE_tensorflow/blob/master/image/result3.png)

## Reference Papers
**1. https://arxiv.org/abs/1312.6114**

**2. https://arxiv.org/abs/1606.05908**

## References

**1.https://github.com/hwalsuklee/tensorflow-mnist-VAE**

**2.https://github.com/shaohua0116/VAE-Tensorflow**

**3.http://cs231n.stanford.edu/**

**4.https://www.facebook.com/groups/TensorFlowKR/permalink/496009234073473/?hc_location=ufi**

**-- Above Reference is ppt which is distributed by Hwal-Suk Lee from facebook page tensorflow korea**

**5.http://jaejunyoo.blogspot.com/2017/04/auto-encoding-variational-bayes-vae-1.html**
