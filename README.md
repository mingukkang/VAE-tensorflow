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
def gaussian_encoder(X, n_hidden, n_z, keep_prob):
    w_init = tf.contrib.layers.xavier_initializer()
    input_shape = X.get_shape()
    
    with tf.variable_scope("encoder_hidden_1", reuse = tf.AUTO_REUSE):
        w1 = tf.get_variable("w1", shape = [input_shape[1], n_hidden], initializer = w_init)
        b1 = tf.get_variable("b1", shape = [n_hidden], initializer = tf.constant_initializer(0.))
        h1 = tf.matmul(X,w1) + b1
        h1 = tf.nn.elu(h1)
        h1 = tf.nn.dropout(h1, keep_prob)
        
    with tf.variable_scope("encoder_hidden_2", reuse = tf.AUTO_REUSE):
        w2 = tf.get_variable("w2", shape = [n_hidden,n_hidden], initializer = w_init)
        b2 = tf.get_variable("b2", shape = [n_hidden], initializer = tf.constant_initializer(0.))
        h2 = tf.matmul(h1,w2) + b2
        h2 = tf.nn.elu(h2)
        h2 = tf.nn.dropout(h2,keep_prob)
    
    with tf.variable_scope("encoder_z", reuse = tf.AUTO_REUSE):
        w3 = tf.get_variable("w3", shape = [n_hidden, n_z*2], initializer = w_init)
        b3 = tf.get_variable("b3", shape = [n_z*2], initializer = tf.constant_initializer(0.))
        h3 = tf.matmul(h2,w3) + b3
        mean = h3[:, : n_z]
        std = tf.nn.softplus(h3[:, n_z :]) + 1e-6
    
    return mean, std
```
**2. Bernoulli Decoder**
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

**3. Variational AutoEncoder**
```python
def Variational_autoencoder(X,n_hidden_encoder,n_z, n_hidden_decoder, keep_prob ):
    X_shape = X.get_shape()
    n_output = X_shape[1]
    
    mean, std = gaussian_encoder(X,n_hidden_encoder, n_z,keep_prob)
    
    z = mean + std*tf.random_normal(tf.shape(mean,out_type = tf.int32), 0, 1, dtype = tf.float32)
    
    X_out = Bernoulli_decoder(z,n_hidden_decoder,n_output,keep_prob)
    X_out = tf.clip_by_value(X_out,1e-8, 1 - 1e-8)
    
    likelihood = tf.reduce_mean(tf.reduce_sum(X*tf.log(X_out) + (1-X)*tf.log(1- X_out),1))
    KL_Divergence = tf.reduce_mean(0.5*tf.reduce_sum(1 - tf.log(tf.square(std) + 1e-8) + tf.square(mean) + tf.square(std), 1))       
    Recon_error = -1*likelihood
    Regularization_error = KL_Divergence
    ELBO = Recon_error + Regularization_error 
    
    return  z ,X_out, Recon_error, Regularization_error, ELBO
```

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
