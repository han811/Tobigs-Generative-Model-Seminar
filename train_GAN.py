import os
import sys
import time

import numpy as np
import pandas as pd

import tensorflow as tf
from model.GAN import Generator, Discriminator, generator_loss, discriminator_loss
from model.config import param

import argparse

gan_generator = Generator()
gan_discriminator = Discriminator()

gen_optimizer = tf.keras.optimizers.Adam(learning_rate=0.002)
disc_optimizer = tf.keras.optimizers.Adam(learning_rate=0.002)

epochs = 100
batch_size = 16


@tf.function
def train_step(inputs):
    latent_vector = tf.random.noraml([batch_size,param['latent_size']])

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
    for epoch in range(epochs):
        gen_loss, disc_loss = train_step(images)
        total_gen_loss += gen_loss
        total_disc_loss += disc_loss
    print('hi')
    
    
if __name__=="__main__":
    train()
    print(os.getcwd())