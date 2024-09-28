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
