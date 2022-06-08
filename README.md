# Transformer
Multi-head Attention Based Transformer Model

## Overview
  * [Intro](#intro)
  * [Requirements](#requirements)
  * [Usage](#usage)
  * [Results](#results)
  * [Dex Swap](#dexswap)
  * [References](#references)


### Intro
  Advancing the deep learning field anew, Transformer networks have become a dominant force to be reckoned with. Taking inspiration from both Gated RNNs and Convolutional Neural Networks, Transformer newtorks intruduced in ["Attention is All You Need"](https://arxiv.org/abs/1706.03762), have brought forth a revolutionary new era for the modeling of long range dynamics. Leveraging a new attention mechanism that encodes positional/temporal alignments, this new architectural structure referred to as Multi-head Attention, is able to project corresponding inputs and outputs into a numerical semantic space that allows for the discovery and preservation of relational characterstics across distances and periods of time that challenged even the most powerful of statistical models. Their use has grown tremendously across the field of machine learning, with application in everything from [language modeling](https://arxiv.org/pdf/1810.04805.pdf), [image generation and classification](https://arxiv.org/pdf/2010.11929.pdf), to [protein folding](https://www.nature.com/articles/d41586-021-03499-y?error=cookies_not_supported&code=80a3df6a-6a29-4845-acb6-1aa735d6e8aa) and [gene expression prediction](https://www.deepmind.com/blog/predicting-gene-expression-with-ai). The Transformer is rapidly tranforming the world around us. Here we demonstrate some of its capabilites with a simple price prediction for ETH/USDC price projections


#### Architecture
The architectural design of the GRU circuit is illustrated below. Each GRU is a node in a deep neueral network that is designed with learnable weight parameters that determine the rate/level of information propogation across a sequence of observed isntances. These units effectively gate the memory of our neural network. They are a condensed and more efficient implementation of the original LSTM gated memory model.

![gru_circuit](images/GRU_circuit.png)

Below is the GRU circuit's math, i.e. the above circuit in the form of math equations whose parameters we will train to estimate

![gur_maths](images/gru_maths.png)


### Requirements
  * Linux distributions that use glibc >= v2.17
  * Python 3.6
  * matplotlib == 3.1.1
  * numpy == 1.19.4
  * pandas == 0.25.1
  * scikit_learn == 0.21.3
  * torch == 1.8.0
  * scipy == 1.7.3


Python pakcage dependencies can be installed using the following command:
```
pip install -r requirements.txt
```
Optional - For training on a GPU (highly recommended), Nvidia CUDA 10.0+ drivers are required

### Usage

Input data is ETH-USDC OHLCV

```
date,closing_price,highest_price,lowest_price,open_price,volume
```

In the command line run to launch an interactive python session with a model that will train on an nvidia gpu if one is available and CUDA is installed, or the CPU if no compatble GPU is found.

```python
python -i train.py
```

### Results

Using our model to predict prices from our last observed data point. Projecting prices 8 minutes into the future. This is a tunable hyperparamter btw, can project further into the future, but at the expense of training time and model accurracy. 


![predicitons_gif](images/animated_graph2.gif)




### DexSwap

dex_swap contains a simple python Brownie script for on-chain conversion of ETH => WETH and then swapped for USDC on uniswap v3


### References

[GRU paper](https://arxiv.org/pdf/1412.3555.pdf)

[GRU applications](https://arxiv.org/pdf/1906.01005.pdf)
