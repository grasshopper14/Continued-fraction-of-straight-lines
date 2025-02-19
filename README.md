 	
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/real-valued-continued-fraction-of-straight/image-classification-on-fashion-mnist)](https://paperswithcode.com/sota/image-classification-on-fashion-mnist?p=real-valued-continued-fraction-of-straight)  
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
