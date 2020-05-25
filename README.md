# S15-DenseNet-for-DepthMap-and-Mask-Image-Prediction-without-using-Pretrained-Model

## Introduction ## 
## Dataset Statistics from Previous Assignment: ##

**Here we are going to have a dataset module, where already creation part done as Session 14/15 A Assignment.** <br>
**Bg Images:** 100 

**Fg Images:** 100 

**Fg Masks:** 100

**Bg_Fg Images:** 400000

**Bg_Fg Images Masks:** 400000

**Bg_Fg Images DepthMap:** 400000


|**Dataset:**|Link|
|------------|----|
|**Dataset Colab (pynb):**|https://colab.research.google.com/drive/1gVyUY93azAIvZVuts5Pm1J1WG76rYgoA|
|**Github link:**|https://github.com/gudaykiran/EVA-Session-14-15A/blob/master/Session15A_Dataset_Creation.ipynb|
|**DepthMap Creation (pynb) Colab:**|https://colab.research.google.com/drive/1BvpvWvAAWcUBBtRws20h5am1DiLQsTG3|
|**Github Link:**|https://github.com/gudaykiran/EVA-Session-14-15A/blob/master/Session_15A_Depthmaps_Dataset_Creation.ipynb|


## Dataset and Data Loaders ##
**Dataset:** Here we will construct a Dataset class which takes input of all 4 images, where to the Network Passing Background and Foreground- Background images as Input to Convolutional Blocks. Target images are Masks and Depth Images. <br>

**Transformations:** We will apply some scale transformations before loading the data.
As both the inputs are of 224 * 224 size, we will resize it as 128 * 128 or 64 * 64
-Grayscale Transformations has been done. <br>

**Dataloaders:** Here we will fetch images in batches apply transforms to them & then returns dataloders for train and validation phases.

**Dataset Utilities:** https://github.com/gudaykiran/S15-DenseNet-for-DepthMap-and-Mask-Image-Prediction-without-using-Pretrained-Model/tree/master/utils <br>

## Mask and DepthMap Images Prediction using Dense CNN Architecture ##
![Mask and DepthMap Images Prediction using Dense CNN Architecture](https://github.com/gudaykiran/S15-DenseNet-for-DepthMap-and-Mask-Image-Prediction-without-using-Pretrained-Model/blob/master/DenseNet%20for%20DepthMap%20and%20Mask%20Image%20Prediction%20without%20using%20Pretrained%20Model.png)


**Architecture Description**  <br>
![Description](https://github.com/gudaykiran/S15-DenseNet-for-DepthMap-and-Mask-Image-Prediction-without-using-Pretrained-Model/blob/master/DNN%20Description.png) 

**DenseNet Py  File:** https://github.com/gudaykiran/S15-DenseNet-for-DepthMap-and-Mask-Image-Prediction-without-using-Pretrained-Model/blob/master/DNN%20Model/net.py

**Number of Parameters : 1977280**

**Loss functions:**

We are using BCE Loss function and SSIM Loss function.

**SSIM Loss function:**
Default loss function in encoder-decoder based image reconstruction had been L2 loss. Previously, Caffe only provides L2 loss as a built-in loss layer. Generally, L2 loss makes reconstructed image blurry because minimizing L2 loss means maximizing log-likelihood of Gaussian. As you know Gaussian is unimodal.

L1 gains a popularity over L2 because it tends to create less blurry images. However, using either L1 or L2 loss in learning takes enormous time to converge. Both losses are pointwise, error is back-propagated by pixel by pixel.

Recently have discovered using SSIM Loss in github for image restructuring:

https://github.com/arraiyopensource/kornia

SSIM loss compares local region of target pixel between reconstructed and original images, whereas L1 loss compares pixel by pixel.

I compare perceptual loss and perceptual loss + SSIM loss in reconstruction of images. We can see perceptual + SSIM loss outperforms only perceptual loss. You can inspect more about SSIM in neural network field in Arxiv. 

**Basic Usage of Loss Function**

 import pytorch_ssim <br>
import torch <br>
from torch.autograd import Variable <br>

img1 = Variable(torch.rand(1, 1, 256, 256)) <br>
img2 = Variable(torch.rand(1, 1, 256, 256)) <br>

if torch.cuda.is_available(): <br>
    img1 = img1.cuda() <br>
    img2 = img2.cuda() <br>
 
print(pytorch_ssim.ssim(img1, img2)) <br>

ssim_loss = pytorch_ssim.SSIM(window_size = 11) <br>

print(ssim_loss(img1, img2)) <br>


SSIM is a built in method part of Sci-Kit’s image library so we can just load it up. 

MSE will calculate the mean square error between each pixels for the two images we are comparing. Whereas SSIM will do the opposite and look for similarities within pixels; i.e. if the pixels in the two images line up and or have similar pixel density values. 

The only issues are that MSE tends to have arbitrarily high numbers so it is harder to standardize it. While generally the higher the MSE the least similar they are, if the MSE between picture sets differ appears randomly, it will be harder for us to tell anything. SSIM on the other hand puts everything in a scale of -1 to 1 (but I was not able to produce a score less than 0). A score of 1 meant they are very similar and a score of -1 meant they are not similar. 

SSIM was already imported through skimage, no need to manually code it. Now we create a function that will take in two images, calculate it’s mse and ssim and show us the values all at once.different. In my opinion this is a better metric of measurement.


**Binary Cross Entropy – Logit Loss Function/ BCE Loss Function:**

If we fit a model to perform this classification, it will predict a probability of being green to each one of our points. Given what we know about the color of the points, how can we evaluate how good (or bad) are the predicted probabilities? This is the whole purpose of the loss function! It should return high values for bad predictions and low values for good predictions. 

For a binary classification like our example, the typical loss function is the binary cross-entropy / log loss.nn.BCEwithLogitsLoss – this loss combines a sigmoid layer and the BCELoss in one single class.


![Binary Cross Entropy – Logit Loss Function/ BCE Loss Function](https://github.com/gudaykiran/S15-DenseNet-for-DepthMap-and-Mask-Image-Prediction-without-using-Pretrained-Model/blob/master/Loss%20Function.png)

## Training the Model ##

Here the input images i.e Bg images and BgFg images are passed through forward propagation and loss functions as Criterion 1 and Criterion 2.

Computing Loss between (Output[1], Mask images) as Criterion1 Loss function and Computing Loss between (Output[2], Depth images) as Criterion 2 Loss function.

Overall loss is calculated as 2 * loss 1 + loss 2 

**Inferences and Showing Outputs as 5 Layers:**

1.	Loss 1 of trained model (epochs wise when loss gets decreased this layer will be shown black)
2.	Loss 2 of trained model (epochs wise when loss gets decreased this layer will be shown black)
3.	Predicted Mask Images by obtaining overall loss
4.	Predicted Depth Images by obtaining overall loss
5.	Predicted BgFg Images by obtaining overall loss

## Final Output ##
![](https://github.com/gudaykiran/S15-DenseNet-for-DepthMap-and-Mask-Image-Prediction-without-using-Pretrained-Model/blob/master/Final%20Output.png)

## Colab Files ##

|File|Link|
|----|----|
|**1. Session15b_Model2-sample1.pynb :**|https://colab.research.google.com/drive/1cd-LFUQBL8WHhUR8NLnsNQ-7nHM5_Ph0|
|**2. Session15b_Model2-sample2.pynb :**|https://colab.research.google.com/drive/1KjChjxxK10Fe8SJEf8GOmWjT0yhqW5es|
|**3. Session15b_Model2-sample3.pynb :**|https://colab.research.google.com/drive/1QsXOYaglO8OfrURD0bmir3KMzWANIwI0|
|**4. Session15b_Model2-sample4.pynb :**|https://colab.research.google.com/drive/14QpNLlsikAOBLrj0qltQEyps0gnrO3gd|


***Submitted by : G Uday Kiran***
