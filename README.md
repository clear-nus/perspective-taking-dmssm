# Latent Emission-Augmented Perspective-Taking (LEAPT) for Human-Robot Interaction
This repository contains the code for our paper [Latent Emission-Augmented Perspective-Taking (LEAPT) for Human-Robot Interaction](https://arxiv.org/pdf/2308.06498.pdf) (IROS-23).

## Introduction

Perspective-taking is the ability to perceive or understand a situation or concept from another individual's point of view, and is crucial in daily human interactions. Enabling robots to perform perspective-taking  remains an unsolved problem; existing approaches that use deterministic or handcrafted methods are unable to accurately account for uncertainty in partially-observable settings. This work proposes to address this limitation via a deep world model that enables a robot to perform both perception and conceptual perspective taking, i.e., the robot is able to infer what a human sees and believes. The key innovation is a decomposed multi-modal latent state space model able to generate and augment fictitious observations/emissions. Optimizing the ELBO that arises from this probabilistic graphical model enables the learning of uncertainty in latent space, which facilitates uncertainty estimation from high-dimensional observations. We tasked our model to predict human observations and beliefs on three partially-observable HRI tasks. Experiments show that our method significantly outperforms existing baselines and is able to infer visual observations available to other agent and their internal beliefs. 
<p align="center">
  <img src="https://github.com/clear-nus/perspective-taking-dmssm/blob/main/image/LEAPT.png?raw=true" width="50%">
  <br />
  <span>Fig 1. Perspective-Taking Example. (A) As the robot (green) is holding the table, it cannot see the peg and hole (purple), which is clearly visible to the human (light blue). The robot considers the perspective of the human and reasons about what the human can observe, despite not knowing what he actually sees. (B) Human belief inference. In brief, the robot's world model (self-model) is used to sample possible environment states and predict human observations, and the human model is used to infer human belief. By leveraging the inference network and perspective taking-model, we can infer human belief under each sampled environment state.</span>
</p>

## Environment Setup 

The code is tested on Ubuntu 20.04, Python 3.8+ and CUDA 11.4. Please download the relevant Python packages by running:

Get dependencies:

```
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip3 install numpy
```

## BibTeX

To cite this work, please use:

```
@article{chen2023latent,
  title={Latent Emission-Augmented Perspective-Taking (LEAPT) for Human-Robot Interaction},
  author={Chen, Kaiqi and Lim, Jing Yu and Kuan, Kingsley and Soh, Harold},
  booktitle={2022 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)},
  year={2023}
}
```
