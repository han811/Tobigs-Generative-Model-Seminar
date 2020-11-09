import tensorflow as tf # TF 2.X
from config import param
'''
    generator
'''
class Generator(tf.keras.Model):
    def __init__(self):
        super(Generator, self).__init__()
        
        self.layer1 = tf.keras.layers.Dense(param['Gen_param']['g_layer1']['units'], activation=tf.nn.relu)
        self.layer2 = tf.keras.layers.Dense(param['Gen_param']['g_layer2']['units'], activation=tf.nn.relu)
        self.layer3 = tf.keras.layers.Dense(param['Gen_param']['g_layer3']['units'], activation=tf.nn.relu)
        self.layer4 = tf.keras.layers.Dense(param['Gen_param']['g_layer4']['units'], activation=tf.nn.sigmoid)
        self.reshape = tf.keras.layers.Reshape((28,28,1))
    
    def call(self, inputs):
        return self.reshape(self.layer4(self.layer3(self.layer2(self.layer1(inputs)))))
        
'''
    discriminator
'''
class Discriminator(tf.keras.Model):
    def __init__(self):
        super(Discriminator, self).__init__()
        
        self.flatten = tf.keras.layers.Flatten()
        self.layer1 = tf.keras.layers.Dense(param['Dis_param']['d_layer1']['units'], activation=tf.nn.relu)
        self.layer2 = tf.keras.layers.Dense(param['Dis_param']['d_layer2']['units'], activation=tf.nn.relu)
        self.layer3 = tf.keras.layers.Dense(param['Dis_param']['d_layer3']['units'], activation=tf.nn.relu)
        self.layer4 = tf.keras.layers.Dense(param['Dis_param']['d_layer4']['units'], activation=tf.nn.softmax)
        
    def call(self, inputs):
        return self.layer4(self.layer3(self.layer2(self.layer1(self.flatten(inputs)))))

'''
    generator loss
'''
def generator_loss(generated_img):
    tf.keras.losses.BinaryCrossentropy()


'''
    discriminator loss
'''
    
if __name__=='__main__':
    gen = Generator()
    dis = Discriminator()
    
    gen_input = tf.keras.Input(shape=(2))
    dis_input = tf.keras.Input(shape=(28,28,1))
    
    gen_model = tf.keras.Model(inputs=gen_input,outputs=gen(gen_input))
    dis_model = tf.keras.Model(inputs=dis_input,outputs=dis(dis_input))
                               
    print(gen_model.summary())
    print(dis_model.summary())
    
    