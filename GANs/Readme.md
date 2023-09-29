# GANs:

## DC-GAN: Deep Convolutional GAN
## TGAN: Temporal Generative Adversarial Nets 
[[paper]](https://openaccess.thecvf.com/content_ICCV_2017/papers/Saito_Temporal_Generative_Adversarial_ICCV_2017_paper.pdf)
     Temporal Generative Adversarial Nets with Singular Value Clipping (Saito, Matsumoto, & Saito, 2017)
## WGAN: Wasserstein GAN
   [[Lipschitz continuity]](https://en.wikipedia.org/wiki/Lipschitz_continuity)
## WGAN-GP: Wasserstein GAN with Gradient Penalty
The first paper is the original WGAN paper and the second proposes GP (as well as weight clipping) to WGAN in order to enforce 1-Lipschitz continuity and improve stability.

Wasserstein GAN (Arjovsky, Chintala, and Bottou, 2017): [[paper]](https://arxiv.org/abs/1701.07875)

Improved Training of Wasserstein GANs (Gulrajani et al., 2017): [[paper]](https://arxiv.org/abs/1704.00028)

This article provides a great walkthrough of how WGAN addresses the difficulties of training a traditional GAN with a focus on the loss functions.

From GAN to WGAN (Weng, 2017): [[paper]](https://lilianweng.github.io/lil-log/2017/08/20/from-GAN-to-WGAN.html)

## SN-GAN: Spectrally Normalized Generative Adversarial Networks
Spectrally Normalized Generative Adversarial Networks:  [[paper]](https://arxiv.org/abs/1802.05957)

As its name suggests, SN-GAN normalizes the weight matrices in the discriminator by their corresponding spectral norm, which helps control the Lipschitz constant of the discriminator. As you have learned with WGAN, Lipschitz continuity is important in ensuring the boundedness of the optimal discriminator. In the WGAN case, this makes it so that the underlying W-loss function for the discriminator (or more precisely, the critic) is valid.

As a result, spectral normalization helps improve stability and avoid vanishing gradient problems, such as mode collapse.

## Conditional GAN
Conditional Generative Adversarial Nets(Mirza and Osindero, 2014) [[paper]](https://arxiv.org/abs/1411.1784)

## InfoGAN
to generate disentangled outputs, based on the paper: InfoGAN: Interpretable Representation Learning by Information Maximizing Generative Adversarial Nets 
 by Chen et. al. [[paper]](https://arxiv.org/abs/1606.03657)
 
 While there are many approaches to disentanglement, this is one of the more widely used and better known.

InfoGAN can be understood like this: you want to separate your model into two parts: 𝑧, corresponding to truly random noise, and 𝑐 corresponding to the "latent code." The latent code 𝑐 which can be thought of as a "hidden" condition in a conditional generator, and you'd like it to have an interpretable meaning. 

Now, you'll likely immediately wonder, how do they get 𝑐 , which is just some random set of numbers, to be more interpretable than any dimension in a typical GAN? The answer is "mutual information": essentially, you would like each dimension of the latent code to be as obvious a function as possible of the generated images. Read on for a more thorough theoretical and practical treatment.

## Controlable GAN
controlling GAN generations using latent space

Interpreting the Latent Space of GANs for Semantic Face Editing (Shen, Gu, Tang, and Zhou, 2020) [[paper]](https://arxiv.org/abs/1907.10786)


# Evaluation
##  Fréchet distance
"The Fréchet distance between multivariate normal distributions" by Dowson and Landau (1982), https://core.ac.uk/reader/82269844

## Fréchet Inception Distance (FID)
Fréchet Inception Distance (Jean, 2018): https://nealjean.com/ml/frechet-inception-distance/

## Inception Score (IS)
FID has overtaken Inception Score (IS)? This paper illustrates the problems with using Inception Score.

A Note on the Inception Score (Barratt and Sharma, 2018): https://arxiv.org/abs/1801.01973

GAN — How to measure GAN performance? (Hui, 2018): https://medium.com/@jonathan_hui/gan-how-to-measure-gan-performance-64b988c47732

## Precison & Recall
Improved Precision and Recall Metric for Assessing Generative Models (Kynkäänniemi, Karras, Laine, Lehtinen, and Aila, 2019): https://arxiv.org/abs/1904.06991

## Perceptual path length (PPL)
 PPL was a metric that was introduced as part of StyleGAN [[]paper](https://arxiv.org/abs/1812.04948) to evaluate how well a generator manages to smoothly interpolate between points in its latent space. In essence, if you travel between two points images produced by a generator on a straight line in the latent space, it measures the total "jarringness" of the interpolation when you add together the jarringness of each step. In this notebook, you'll walk through the motivation and mechanism behind PPL.

The StyleGAN2 [[paper]](https://arxiv.org/abs/1912.04958) noted that metric also "correlates with consistency and stability of shapes," which led to one of the major changes between the two papers.

## Perceptual Similarity
Like FID, PPL uses the feature embeddings of deep convolutional neural network. Specifically, the distance between two image embeddings as proposed in The Unreasonable Effectiveness of Deep Features as a Perceptual Metric by Zhang et al (CVPR 2018) [[paper]](https://arxiv.org/abs/1801.03924). In this approach, unlike in FID, a VGG16 network is used instead of an InceptionNet.

Perceptual similarity is closely similar to the distance between two feature vectors, with one key difference: the features are passed through a learned transformation, which is trained to match human intuition on image similarity. Specifically, when shown two images with various transformations from a base image, the LPIPS ("Learned Perceptual Image Patch Similarity") metric [[code]](https://github.com/richzhang/PerceptualSimilarity) is meant to have a lower distance for the image that people think is closer. 


# Bias
Machine Bias (Angwin, Larson, Mattu, and Kirchner, 2016): https://www.propublica.org/article/machine-bias-risk-assessments-in-criminal-sentencing




# Some databases
* MNIST Database: http://yann.lecun.com/exdb/mnist/
* CelebFaces Attributes Dataset (CelebA):  is a dataset of annotated celebrity images, http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html
* ImageNet: http://www.image-net.org/






# Some references
Deconvolution and Checkerboard Artifacts (Odena et al., 2016) :  http://doi.org/10.23915/distill.00003

Wasserstein GAN (Arjovsky, Chintala, and Bottou, 2017): https://arxiv.org/abs/1701.07875

Improved Training of Wasserstein GANs (Gulrajani et al., 2017): https://arxiv.org/abs/1704.00028

StyleGAN - Official TensorFlow Implementation: https://github.com/NVlabs/stylegan

Stanford Vision Lab: http://vision.stanford.edu/

Review: Inception-v3 — 1st Runner Up (Image Classification) in ILSVRC 2015 (Tsang, 2018): https://medium.com/@sh.tsang/review-inception-v3-1st-runner-up-image-classification-in-ilsvrc-2015-17915421f77c

HYPE: A Benchmark for Human eYe Perceptual Evaluation of Generative Models (Zhou et al., 2019): https://arxiv.org/abs/1904.01121

Improved Precision and Recall Metric for Assessing Generative Models (Kynkäänniemi, Karras, Laine, Lehtinen, and Aila, 2019): https://arxiv.org/abs/1904.06991

Large Scale GAN Training for High Fidelity Natural Image Synthesis (Brock, Donahue, and Simonyan, 2019): https://arxiv.org/abs/1809.11096

The Fréchet Distance between Multivariate Normal Distributions (Dowson and Landau, 1982): https://core.ac.uk/reader/82269844




# HYPE
intrigued about human evaluation and HYPE (Human eYe Perceptual Evaluation) of GANs? Learn more about this human benchmark in the paper! 

HYPE: A Benchmark for Human eYe Perceptual Evaluation of Generative Models (Zhou et al., 2019): https://arxiv.org/abs/1904.01121




