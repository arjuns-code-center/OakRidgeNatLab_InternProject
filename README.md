# ORNL_Coding
Summer 2021 Internship with the Oak Ridge National Laboratory. Mentored by Dr. Junqi Yin.

Traditional Machine Learning (ML) is well adopted for the analysis of scientific datasets. These methods include Principal Component Analysis (PCA), which reduces data dimensions through the use of statistics and linear algebra, and K-Means clustering, which clusters data as distinctly as possible through iterative spatial optimization. This adoption can further be accelerated through the use of High Performance Computers (HPCs). 

ML has recently attracted a lot of interest in science due to the flexibility it offers scientists. With most ML cases, the ground truth is not known, which prompts the use of unsupervised ML algorithms. Traditionally, PCA and K-Means have been widely used, but it becomes challenging to maintain accuracy with them in large feature datasets with strong non-linearity. 

This research looks at the differences between four methods: PCA, K-Means, a Convolutional Autoencoder (CAE), and a Convolutional Variational Autoencoder (CVAE). The two case studies used for testing are the SARS-MERS-COVID dataset modelling the three viruses and their contact maps at different samples, and a High Entropy Alloy (HEA) dataset modelling different configurations of a HEA at different given temperatures. A HEA is a combination of multiple elements in equal amounts. There has been significant interest in HEAs recently because they are designable materials with wide applications, like their use in plane engines due to their high temperature and corrosion resistance. 

We find that there is greater accuracy with the CAE and CVAE when it comes to large feature space. In addition to comparing the accuracy, we also compare the time it takes for each of these methods to execute completely at different numbers of CPUs, GPUs, and the complexity of input data. We find that CAE and CVAE have more cost of execution but provide greater accuracy to compensate for it, while PCA and K-Means have minimal execution costs with reduced accuracy as the feature space gets larger on both case studies.

You can find in this repository the code I have written for each of the above mentioned algorithms as well as test results I got when I ran it through the Summit supercomputer. The .ipynb in each folder shows plotting results I received after testing the code. 
