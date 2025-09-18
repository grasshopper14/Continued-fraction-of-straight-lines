 	
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/real-valued-continued-fraction-of-straight/image-classification-on-fashion-mnist)](https://paperswithcode.com/sota/image-classification-on-fashion-mnist?p=real-valued-continued-fraction-of-straight)  

Image classification of Fashion-MNIST dataset with just 7860 parameters - A state of the art technique in terms of number of trainable parameters as observed on paperswithcode [dot] com on April 2025. 

Continued fraction of straight line equation $y=mx$
has shown significant improvement in convergence of logistic regression by inherently making gradient descent stepping adaptive: https://arxiv.org/abs/2412.16191 This technique can provide results without regularization in machine learning.

These are two-parameter superposable S-curves posed as a univarsal distribution function: https://arxiv.org/abs/2504.19488

Here, the continued fraction is just an interpretation of the formulation. And this is the best way to address nonlinear behavior of various quantities of interest.

Enjoy fast convergence without the knowledge of "Regularization" 
and with step size set to unity.
All you need to do is do some algebraic manipulation as presented in:
https://arxiv.org/abs/2412.16191

This algebraic manipulation can also be interpreted as a continued fraction of straight lines.

The accuracy obtained using this method on the Fashion-MNIST test daatset is 0.84.
This can be compared with the following accuracies presented in the Fashion-MNIST paper:

LogisticRegression  
C=1 multi_class=ovr penalty=l1 0.842  
C=1 multi_class=ovr penalty=l2 0.841  
C=10 multi_class=ovr penalty=l2 0.839  
C=10 multi_class=ovr penalty=l1 0.839  
C=100 multi_class=ovr penalty=l2 0.836  
  
 This interpretation also explains sigmoidal growth curves in biological sciences as infinitely regulated linear time scale: https://doi.org/10.1101/2025.01.07.631841  

And any nonlinear curve can be written as a sum of two sigmoidal curves: https://doi.org/10.1101/2025.02.06.636984  

Continued fraction of straight lines lead to sigmoidal curves. When these curves are superposed, highly nonlinear fitting of data is possible. This has a widespread application in biological growth, machine learning techniques, antibiotic resistance research: https://doi.org/10.1101/2025.01.27.634991 and so on.
