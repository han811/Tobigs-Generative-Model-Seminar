'''
Generator
'''
g_layer1 = dict()
g_layer1['units'] = 128
g_layer1['activation'] = 'relu'

g_layer2 = dict()
g_layer2['units'] = 256
g_layer2['activation'] = 'relu'

g_layer3 = dict()
g_layer3['units'] = 512
g_layer3['activation'] = 'relu'

g_layer4 = dict()
g_layer4['units'] = 28 * 28 * 1
g_layer4['activation'] = 'tanh'

Gen_param = {}
Gen_param['g_layer1'] = g_layer1
Gen_param['g_layer2'] = g_layer2
Gen_param['g_layer3'] = g_layer3
Gen_param['g_layer4'] = g_layer4

'''
Discriminator
'''
d_layer1 = dict()
d_layer1['units'] = 512
d_layer1['activation'] = 'relu'

d_layer2 = dict()
d_layer2['units'] = 256
d_layer2['activation'] = 'relu'

d_layer3 = dict()
d_layer3['units'] = 128
d_layer3['activation'] = 'relu'

d_layer4 = dict()
d_layer4['units'] = 2
d_layer4['activation'] = 'None'

Dis_param = {}
Dis_param['d_layer1'] = d_layer1
Dis_param['d_layer2'] = d_layer2
Dis_param['d_layer3'] = d_layer3
Dis_param['d_layer4'] = d_layer4


param = {}
param['Gen_param'] = Gen_param
param['Dis_param'] = Dis_param
param['latent_size'] = 2