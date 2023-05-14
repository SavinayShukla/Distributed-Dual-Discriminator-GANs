# Distributed Training of Dual Discriminator GANs.
### Authors : Savinay Shukla(ss16924), Ishaan Pitty (ip2244) 


<br>

In this project, we propose add another discriminator in a typical DCGAN training pipeline for the Generator to acheive better and faster convergence. Adding another discriminator will force the Generator to fool two discriminators at once, and disallow the dominance of Discriminator training. 

Our results are based on implementing our improvised training pipeline across various datasets which include **CIFAR-10, CIFAR-100, MNIST and SVHN**. We benchmarked our performance on various datasets to know possible limitations and improvements across a diverse set of images.

GPUs used for single-node and distributed-training : **NVIDIA RTX 3070, NVIDIA Tesla V100**.

**Copy of presentation**: https://tome.app/n-011/double-the-discrimination-clhmm4xfw1vyncd40wpg6enqe

![Alt text](plots/paperImage.PNG)


Training Pipeline:


![Alt text](plots/Diagram.gif)

## Generation Demo

The demo on the **left is for DCGAN** image generation on the MNIST dataset, and **right one is our implementation**. As you can see, we generalize faster, and better.

<p align="center">
<img src="plots/Dc.gif">
<img src="plots/D2.gif">
</p>


## How to run?

Our proposed implementation (both single node and distributed) are in the folder `D3GAN`. Run the following command for **single GPU** training: 

```
python3 train.py
```

and for **multi GPU** training:

```
python3 train_ddp.py
```

Configurations can be set in the `config.ini` file, for different hyper-parameters. We have selected some parameters for you, which according to us produces the most consistent comparision.

Same can be done for `DCGAN`.

**Please note**: DCGAN implementation is for benchmarking only.

# Results and Observations

We want to emphasis on the evaluation metrics for generative models, and how generator loss per iteration, could lead to false interpretations of the quality of images.
We observed our implementation against the state of the art DCGAN model for both **Inception Scores** and **Frechet Inception Distance**.
Below are the plots for how fast our implementation converges for different batch sizes.
| ![Alt text](plots/FID_score_comparision_64.png) | 
|:--:| 
| FID Score: CIFAR-10 for Batch Size - 64 |

| ![Alt text](plots/IS_score_comparision.png) | 
|:--:| 
| IS Score: CIFAR-10 for Batch Size - 64 |

| ![Alt text](plots/FID_score_comparision_256.png) | 
|:--:| 
| FID Score: CIFAR-10 for Batch Size - 256 |

| ![Alt text](plots/FID_score_comparision_512.png) | 
|:--:| 
| FID Score: CIFAR-10 for Batch Size - 512 |

# Distributed Analysis

Below are some benchmarks on the scalibility of our implementation across multiple GPUs.

| ![Alt text](plots/ddp_throughput.png) | 
|:--:| 
| Throughput Comparision |

| ![Alt text](plots/ddp_traiing_time.png) | 
|:--:| 
| Training Time Comparision |

| ![Alt text](plots/ddp_utilization.png) | 
|:--:| 
| GPU Utilization |






