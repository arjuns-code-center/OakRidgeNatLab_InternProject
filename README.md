# Evaluation of Unsupervised Machine Learning Methods on Scientific Datasets in High Performance Computing
### An Oak Ridge National Lab summer intern project.
### Mentored by Dr. Junqi Yin

# Introduction
Scientists who perform High Performance Computing (HPC) often run traditional Machine Learning (ML) methods on data. These methods include Principal Component Analysis (PCA) and K-Means clustering. ML has recently attracted a lot of interest in science due to the flexibility it offers scientists. With most ML cases, the ground truth is not known, which prompts the use of unsupervised ML algorithms. 

Traditionally, PCA and K-Means have been widely used, but it becomes challenging to maintain accuracy with them in large feature datasets with strong non-linearity. This research looks at the differences between four methods: PCA, K-Means, a Convolutional Autoencoder (CAE) and a Convolutional Variational Autoencoder (CVAE). The two case studies used for testing are the SARS-MERS-COVID dataset and a High Entropy Alloy (HEA) dataset. We find that there is greater accuracy with the CAE and CVAE when it comes to large datasets. We also find that CAE and CVAE provide greater accuracy at greater cost in computation compared to PCA and K-Means.

# Datasets
We used pathogen strains of SARS, MERS, and COVID as images to train the different models. The aim is the use the strains to classify each strain accurately. This dataset consists of `60000` samples and each sample is an image that is `24x24x1` pixels. 

![image](https://github.com/user-attachments/assets/90d39276-ac69-4fcd-a5ee-6ab56c48911c)

The second dataset we used in the High Entropy Allow dataset. This dataset consists of lattice structures of different alloys. This dataset has dimensions of `40000` samples, each `40x40x40` pixels. 

![image](https://github.com/user-attachments/assets/c7b0a22b-f855-4523-8ce4-60063d800774)

# Methods
We use PCA, KMeans, a Convolutional Autoencoder, and a Convolutional Variational Autoencoder as the primary methods for evaluating this dataset. Each method is elaborated below. 

## Principal Component Analysis (PCA):
PCA is a dimensionality reduction method that exposes the most important features from the least important ones in a list. Using this method, it is possible to expose the most impotant features from the datset and use them, especially when there are a lot of features into the models. This is especially useful when there are a lot of features, and some of them are dependent on each other, or some of them are hurting the model performance. Shown below is the flow chart representing PCA algorithm. 

![image](https://github.com/user-attachments/assets/143438e7-0f0b-4e88-94d6-5413a089db52)

## K-Means Clustering:
K-Means is a way to cluster different categories together based on centroids and distances to those centroids. It is an iterative procedure that starts with randomly assigned centroids, takes repetitive means and distances for every point, and assigns that to the closest centroid. This algorithm eventually converges, revealing the final clusters. Shown below is a flow chart of the algorithm. 

![image](https://github.com/user-attachments/assets/722e3d8e-cd1c-43f8-b36a-1fc98833f969)

## Convolutional Autoencoder
Autoencoders shrink an image, and use the data points found within the image while expanding it back up again to learn intrinsic properties about the image. This helps the model get really good at classifying images based on how similar it is from learned features while shrinking and expanding trial images over and over again. A convolutional autoencoder passes a kernel over the pixel values of the image, and creates a smaller mappingl, effectively shrinking the image dimensions (linear algebra). The CNN model is shown below, as well as a representation of how it works. 

![image](https://github.com/user-attachments/assets/58e70c5a-1698-4633-b5b5-b514e948cb47)

## Convolutional Variational Autoencoder
Just like the way a CAE works, CVAE introduces a probabilistic manner of describing features in latent space. It outputs probability distributions instead of a single value, modelling uncertainty to a greater degree. So we randomly sample from each probability distribution to generate decoder output. So then the decoder attempts to reconstruct the image based on the sampling done. This helps with greater learning given enough training data, making CVAE more generalizable to a variety of testing data. 

![image](https://github.com/user-attachments/assets/76031e12-f36f-4f6c-8be5-416a183bc355)

## Gaussian Mixture Model
Each cluster is modelled as a Gaussian. The Expectation Maximisation (EM) algorithm is used to maximize the marginal likelihood of the input variables given parameters. We then estimate the posterior distribution conditional on weights, means, and covariances. Once the posterior distribution has been estimated, we obtain the parameters of each Gaussian and evaluate log likelihood. 

![image](https://github.com/user-attachments/assets/ec54ba7a-9747-48f8-8f72-cc39a6c586a2)

## Generative Adversarial Networks
GANs are a great way to train neural networks to classify. It is a 2 part neural network, one of which is a generator and one is a discriminator. The generator learns off the discriminator loss and tries to fool the discriminator into thinking its outputs are real. THe discriminator, however, has to classify between the real and fake inputs. This way, we have an adversarial situation. So the common case becomes that the discriminator is doing really well but the generator is not. The worst case is when the generator is doing really well and the discriminator is not, because now the network is not going to discriminate between real and fake inputs. 

![image](https://github.com/user-attachments/assets/898f82ae-0561-4fdd-b51e-56058dede37f)

# Results
We found that the different algorithms we used here gave us different classification outputs. For example, K-Means was not very good at classifying on the HEA dataset, but the CAE and CVAE did really well. On the SARS-MERS-COVID dataset, a neural network trained with PCA reduction did really well, achieving 74% accuracy, and K-Means also did not do so well, achieving only 56.87% accuracy. The results are shown below. 

## PCA
![image](https://github.com/user-attachments/assets/0f01ae24-5d0d-46ed-92aa-3b7813f70a0a) ![image](https://github.com/user-attachments/assets/dfa1992c-3efd-4d32-97e7-aad001ebf8f4)

## K-Means
![image](https://github.com/user-attachments/assets/8ac2703e-3008-4d2b-8d54-550b9f5f6b3b) ![image](https://github.com/user-attachments/assets/e32d0932-e44c-4368-9bf4-3a74ab2c8b27)

## CAE
![image](https://github.com/user-attachments/assets/a9d064fe-9929-48c0-bef7-9a21ff79f99a) ![image](https://github.com/user-attachments/assets/82c5f18e-6f1f-47a5-b267-7ffe883daf60)

## CVAE
![image](https://github.com/user-attachments/assets/9ea1cf79-fa8d-473d-be26-3f800caaccce) ![image](https://github.com/user-attachments/assets/fda24cb1-6bb6-4974-bec9-253e04ee9b40)

## GMM
![image](https://github.com/user-attachments/assets/dff07b5d-9bbb-4c1a-9533-8d43652d6c45) ![image](https://github.com/user-attachments/assets/f4b53272-d52f-4abf-b887-208256ca57a1)

# Multi-GPU Training
We were also able to train across multiple GPUs for these models on the Summit supercomputer. To do this in code, it involved using the Horovod python library. Horovod allows us to separate the data or the model and train in parallel. We implemented the data separation across different GPUs, and passed through the whole model on each. We see there is a decrease in processing times from 1 to 6 GPUs used for both datasets. The SARS-MERS-COVID dataset showed greatest decrease in time, whereas the HEA dataset showed less, due to the complexity of the dataset. 

![image](https://github.com/user-attachments/assets/aa9d9660-7b73-4576-a6f6-d8eda89a9937)
