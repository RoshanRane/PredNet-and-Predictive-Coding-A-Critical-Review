# PredNet and Predictive Coding: A Critical Review 
The PredNet architecture by Lotter et al. combines a biologically plausible architecture called Predictive Coding with self-supervised video prediction in order to learn the complex structure of the visual world. While the architecture has drawn a lot of attention and various extensions of the model exist, there is a lack of a critical analysis. We fill in the gap by evaluating PredNet, both as an implementation of Predictive Coding theory and as a self-supervised video prediction model, using a challenging video action classification dataset. We also design an extended architecture to test if conditioning future frame predictions on the action class of the video and vise-versa improves the model performance. With substantial evidence, we show that PredNet does not completely follow the principles of Predictive Coding. Our comprehensive analysis and results are aimed to guide future research based on PredNet or similar architectures based on the Predictive Coding theory.

<img src="https://github.com/RoshanRane/Predictive-video-classification/blob/master/PredNet_Vanilla.jpg"  height="500" width="750"
     alt="Vanilla PredNet architecture diagram"/>

!["Prednet Architecture"](https://github.com/RoshanRane/Predictive-video-classification/blob/master/PredNet_Vanilla.jpg)
