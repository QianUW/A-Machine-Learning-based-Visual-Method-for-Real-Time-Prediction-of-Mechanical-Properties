# A-Machine-Learning-based-Visual-Method-for-Real-Time-Prediction-of-Mechanical-Properties

# Abstract

Human being can rapidly distinguish objects of surrounding scenarios and further estimate their material properties, only by visual information of shape, lightness, and texture, which kind of ability
can help us understand surrounding environment to make a better decision. However, it is still a
challenge to realize this object perception function for autonomous vehicles or robots. Therefore, this
paper developed a vision-based deep learning method to accurately predict the mechanical properties
of the objects appeared in scenarios. In the beginning, a training set is prepared by adding labels of
elastic modulus and Poisson’s ratio to object categories in an ADE20K training set. A multi-resolution
convolutional neural network is proposed by introducing an information distributor and several detail
extractors, which can significantly reduce computational sources. In addition, the presented method
can predict the mechanical properties of the objects belonging to the unannotated categories, which
does not rely on large-scale annotated training data, and therefore is a weakly-supervised method. By
numerical examples of 2000 images in the validation set, the average relative errors for elastic modulus
is 17% and Poissons ratios is 11%, which indicates that the presented method can accurately estimate
mechanical properties only by application of visual information.

# Paper
For more details, please refer to our [preprint](https://github.com/QianUW/A-Machine-Learning-based-Visual-Method-for-Real-Time-Prediction-of-Mechanical-Properties/blob/master/Preprint%20A%20Vision-based%20Deep%20Learning%20Method%20for%20Real-Time%20Prediction%20of%20Mechanical%20Properties.pdf).

# Samples of Results
# Indoor Samples
![Indoor](Indoor%20Samples.jpg)
# Outdoor Samples
![Outdoor](Outdoor%20Samples.jpg)