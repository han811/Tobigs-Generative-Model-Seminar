import os
import sys
import time

import numpy as np
import pandas as pd

import tensorflow as tf
from model.GAN import Generator, Discriminator

import argparse

@tf.function
def train_step(inputs):
    latent_vector = tf.random.noraml([])

def train():
    
    print('hi')
    
    
if __name__=="__main__":
    train()