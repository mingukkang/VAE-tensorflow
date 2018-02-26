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

![사진2](https://github.com/MINGUKKANG/VAE_tensorflow/blob/master/image/Gaussian_encoder.PNG)

**2. Bernoulli Decoder**

![사진3](https://github.com/MINGUKKANG/VAE_tensorflow/blob/master/image/Bernoulli_decorder.PNG)

**3. Variational AutoEncoder**

![사진4](https://github.com/MINGUKKANG/VAE_tensorflow/blob/master/image/Variational_Autoencoder.PNG)

## Result
```
**1. Comparing the generated images with the original images**

![사진5](https://github.com/MINGUKKANG/VAE_tensorflow/blob/master/image/Result1.PNG)
```
```
**2. Distribution of MNIST Data**

![사진6](https://github.com/MINGUKKANG/VAE_tensorflow/blob/master/image/result2.png)
```
```
**3. Manifold of MNIST Data**

![사진7](https://github.com/MINGUKKANG/VAE_tensorflow/blob/master/image/result3.png)
```

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
