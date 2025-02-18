Enjoy fast convergence without the knowledge of "Regularization" 
and with step size set to unity.
All you need to do is do some algebraic manipulation as presented in:
https://arxiv.org/abs/2412.16191

This algebraic manipulation can also be interpreted as a continued fraction of straight lines.

The accuracy obtained using this method on the Fashion-MNIST test daatset is 0.84.
This can be compared with the following accuracies presented in the Fashion-MNIST paper:

LogisticRegression__
C=1 multi_class=ovr penalty=l1 0.842 __
C=1 multi_class=ovr penalty=l2 0.841__
C=10 multi_class=ovr penalty=l2 0.839 __
C=10 multi_class=ovr penalty=l1 0.839 __
C=100 multi_class=ovr penalty=l2 0.836
