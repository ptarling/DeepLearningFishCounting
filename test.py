import numpy as np
from model import *

def results(count,ypred):
    count_hat = np.sum(ypred[:,:,:,0], axis = (1,2))
    ae = np.abs(count-count_hat)
    se = ((count-count_hat)**2)**0.5
    rmse = (np.mean((count-count_hat)**2))**0.5
    print("                              MAE: ", " {:.2f}".format(np.mean(ae)))
    print("                             RMSE: ", " {:.2f}".format(rmse))


x_val = np.load('/Labelled_data/x_val.npy')
y_val = np.load('/Labelled_data/y_val.npy')
pair1_val = np.load('./pair1_val.npy')
pair2_val = np.load('./pair2_val.npy')

x_test = np.load('Labelled_data/x_test.npy')
y_test = np.load('/Labelled_data/y_test.npy')
pair1_test = np.load('./pair1_test.npy')
pair2_test = np.load('./pair2_test.npy')

#Generate ground truth fish counts from density map
count_test = np.sum(y_test, axis=(1,2))
count_val = np.sum(y_val, axis=(1,2))



    
model = multitask_au(input_shape = (576,320,3))


path_weights = "/weights_mt_au_pretrained.h5" #choose weights to test


model.load_weights(str(path_weights))


counthat_val, diff = model.predict([x_val, pair1_val, pair2_val])
print("val scores:")
results(count_val,counthat_val)
print("")
counthat_test, diff = model.predict([x_test, pair1_test, pair2_test])
print("test scores:")
results(count_test,counthat_test)
