import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import os
import data_loader
import embeds
import model as model_arch
import optimization

# %% Model Hyperparameter
# Device configuration
device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
model_name = 'ETH-USDT_GRU'
# make target OHLC to order column so we can drop V when we create dataset for y_batches
# TODO - testing modeling volume as well so that src matches trt dimensions in transformer inputs
targets = ['closing_price','highest_price','lowest_price','open_price', 'volume']
#targets = ['closing_price','highest_price','lowest_price','open_price'] #must b a list
# Hyper-parameters
sequence_length = 128
pred_seq_length = 8
prediction_frequency = '1min'
num_epochs = 500
batch_size = 100
shuffle_batches = False

# input_dim, hidden_dim, layer_dim, output_dim, dropout_prob)
x_feature_dim = 5 # (OHLCV)
num_hidden_layers = 1
hidden_layer_width = 64
attention_heads = 8
output_dimensions = len(targets)
dropout_probability = 0.3
learning_rate = .1
weight_decay = 1e-6

# GRUModel(x_feature_dim, hidden_layer_width, num_hidden_layers, output_dimensions, dropout_probability).to(device)
print('defined all hyperparams')

#%% Model training specs
# weight initialization
# https://pytorch.org/docs/stable/nn.init.html
# https://stackoverflow.com/questions/49433936/how-to-initialize-weights-in-pytorch/57116138#57116138
def initialize_weights(model_to_initialize):
    for p in model_to_initialize.parameters():
        torch.nn.init.normal_(p, std=(1/(hidden_layer_width)))
    print('random-n inited weights')

# batch_size, sequence_length, input_dim, hidden_dim, layer_dim, nheads, pred_len, output_dim, dropout_prob):
model = model_arch.TransformerModel(
    batch_size= batch_size,
    sequence_length = sequence_length,
    input_dim= x_feature_dim,
    hidden_dim= hidden_layer_width,
    layer_dim= num_hidden_layers,
    nheads= attention_heads,
    pred_len= pred_seq_length,
    output_dim= output_dimensions,
    dropout_prob= dropout_probability).to(device)

initialize_weights(model)
print(model.named_parameters())

criterion = nn.MSELoss()

# Adam
# https://arxiv.org/abs/1412.6980
# optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
# optimal ADAM parameters: Good default settings for the tested machine learning problems are stepsize "lr" = 0.001,
#  Exponential decay rates for the moment estimates "betas" β1 = 0.9, β2 = 0.999 and
#  epsilon decay rate "eps" = 10−8

# AdamW decouple weight decay regularization improving upon standard Adam
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
# https://arxiv.org/abs/1711.05101

#optimizer = torch.optim.LBFGS(model.parameters(), lr=0.08)
# An LBFGS solver is a quasi-Newton method which uses the inverse of the Hessian to estimate the curvature of the
# parameter space. In sequential problems, the parameter space is characterised by an abundance of long,
# flat valleys, which means that the LBFGS algorithm often outperforms other methods such as Adam, particularly when
# there is not a huge amount of data.

# Learning Rate Scheduler
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5,  verbose=True)
# try with Cosine Annealing LR
# torch.optim.lr_scheduler.CosineAnnealingLR

opt = Optimization(model=model, model_name=model_name, loss_fn=criterion, optimizer=optimizer)

opt.train(train_loader, val_loader, batch_size=batch_size, n_epochs=num_epochs, n_features=x_feature_dim)
