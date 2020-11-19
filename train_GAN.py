import os
import sys
import time

import numpy as np
import pandas as pd

from data.mnist import mnist

import tensorflow as tf
from GAN.GAN import Generator, Discriminator, generator_loss, discriminator_loss
from GAN.config import param

import argparse

###############################################
### setting of the arguments and parameters ###
###############################################

parser = argparse.ArgumentParser(description="epoch, batch")
parser.add_argument('--epoch', type=int, default=100, help='epoch size')
parser.add_argument('--batch', type=int, default=64, help='batch size')
parser.add_argument('--data', type=str, default='mnist',help='train data name')
parser.add_argument('--latent_size', default=16, type=int, help='train data name')

args = parser.parse_args()

epochs = args.epoch
batch_size = args.batch
data = args.data
latent_size = args.latent_size

gan_generator = Generator()
gan_discriminator = Discriminator()

gen_optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)
disc_optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)

####################
### data loading ###
####################
def load_data(data_name,shuffle=False):
    data_list = ['mnist','cifar10','celeba']
    if data_name not in data_list:
        print('data name is incorrect!\nplease type correct data name')
        exit()

    if(data_name=='mnist'):
        return mnist(shuffle)
        

@tf.function
def train_step(inputs):
    latent_vector = tf.random.normal([batch_size,latent_size])

    with tf.GradientTape(persistent=True) as tape:
        true_out = gan_discriminator(inputs)
        generated_out = gan_discriminator(gan_generator(latent_vector))

        gen_loss = generator_loss(generated_out)
        disc_loss = discriminator_loss(true_out,generated_out)
    
    grad_gen = tape.gradient(gen_loss, gan_generator.trainable_variables)
    grad_disc = tape.gradient(disc_loss, gan_discriminator.trainable_variables)
    
    gen_optimizer.apply_gradients(zip(grad_gen, gan_generator.trainable_variables))
    disc_optimizer.apply_gradients(zip(grad_disc, gan_discriminator.trainable_variables))
    
    return gen_loss, disc_loss
        
def train():
    train_data, train_label, test_data, test_label = load_data(data,shuffle=True)
    
    for epoch in range(epochs):
        total_gen_loss = 0
        total_disc_loss = 0
        for n_batch in range(batch_size):
            n = int(len(train_data)/batch_size)
            if n_batch==batch_size-1:
                images = train_data[n*n_batch:,:,:]
            else:
                images = train_data[n*n_batch:n*(n_batch+1),:,:]
            gen_loss, disc_loss = train_step(images)
            total_gen_loss += gen_loss
            total_disc_loss += disc_loss
            
        print('generator loss:',total_gen_loss.numpy())
        print('discriminator loss:',total_disc_loss.numpy())
    
    
if __name__=="__main__":
    train()