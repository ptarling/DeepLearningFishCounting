import numpy as np
import tensorflow as tf
from model import *
from lossfunctions_metrics import *
import matplotlib.pyplot as plt
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"



#Load data
x_train = np.load('/Labelled_data/x_train.npy')
y_train = np.load('/Labelled_data/y_train.npy')


ind_shuff = np.random.permutation(len(x_train))
x_train = x_train[ind_shuff]
y_train = y_train[ind_shuff]

x_val = np.load('/Labelled_data/x_val.npy')
y_val = np.load('/Labelled_data/y_val.npy')

pair1_train = np.load('./pair1_train.npy')
pair2_train = np.load('./pair2_train.npy')

pair1_val = np.load('./pair1_val.npy')
pair2_val = np.load('./pair2_val.npy')


rank_train = np.zeros(len(pair1_train))
rank_val = np.zeros(len(pair1_val))


#change shape for variance prediction:
def y_reshape(y_data):
    y_data = np.expand_dims(y_data, -1)
    y_data_reshape = np.concatenate((y_data, np.zeros((y_data.shape))), axis =3)
    return y_data_reshape


y_train_reshape = y_reshape(y_train)
y_val_reshape = y_reshape(y_val)



#Convert all data to tensors
x_train = tf.convert_to_tensor(x_train, dtype = tf.float32)
y_train_reshape = tf.convert_to_tensor(y_train_reshape, dtype = tf.float32)
x_val =  tf.convert_to_tensor(x_val, dtype = tf.float32)
y_val_reshape =  tf.convert_to_tensor(y_val_reshape, dtype = tf.float32)
pair1_train = tf.convert_to_tensor(pair1_train, dtype = tf.float32)
pair2_train = tf.convert_to_tensor(pair2_train, dtype = tf.float32)
pair1_val = tf.convert_to_tensor(pair1_val, dtype = tf.float32)
pair2_val = tf.convert_to_tensor(pair2_val, dtype = tf.float32)
rank_train = tf.convert_to_tensor(rank_train, dtype = tf.float32)
rank_val = tf.convert_to_tensor(rank_val, dtype = tf.float32)




#Build model
model = multitask_au(input_shape = (576,320,3))
#model.load_weights('./weights_mt_au_pretrained.h5') # option to load weights from our trained model
                     
optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
    
@tf.function
def train_on_batch(X,p1,p2,y,diff):
    with tf.GradientTape() as tape:
        yhat,yhat_diff = model([X,p1,p2], training = True)
        yhat = tf.convert_to_tensor(yhat, dtype = tf.float32)
        yhat_diff = tf.convert_to_tensor(yhat_diff, dtype = tf.float32)
        
        ab = au_absolute_loss(y,yhat)
        hinge = hinge_loss(diff,yhat_diff)
       
        loss_value = tf.reduce_sum(ab + hinge)
        
    grads = tape.gradient(loss_value, model.trainable_weights)
    optimizer.apply_gradients(zip(grads,model.trainable_weights))
    return loss_value, hinge, yhat                    
                     
                     
@tf.function
def validate_on_batch(X,p1,p2,y,diff):
    yhat_val,yhat_diff_val = model([X,p1,p2], training = False)
    yhat_val = tf.convert_to_tensor(yhat_val, dtype = tf.float32)
    yhat_diff_val = tf.convert_to_tensor(yhat_diff_val, dtype = tf.float32)
    ab = au_absolute_loss(y,yhat_val)
    hinge = hinge_loss(diff,yhat_diff_val)
       
    loss_value_val = tf.reduce_sum(ab + hinge)
    return loss_value_val, yhat_val
 
    

batch_size = 10
epochs = 200
                     

best_val_mae = 99999
best_val_mae_MA = 99999

all_train_loss = []
all_train_mae = []
all_val_loss = []
all_val_mae = []



for epoch in range(0,epochs):
    train_loss = []
    train_mae = []
    hing = []
                    
    train_data = tf.data.Dataset.from_tensor_slices((x_train, 
                                                     y_train_reshape,
                                                     pair1_train,
                                                     pair2_train,
                                                     rank_train)).shuffle(buffer_size=100).batch(batch_size)
    
    test_data = tf.data.Dataset.from_tensor_slices((x_val,
                                                    y_val_reshape,
                                                    pair1_val,
                                                    pair2_val,
                                                    rank_val)).shuffle(buffer_size=100).batch(batch_size)

    for batch, (X,y,p1,p2,diff) in enumerate(train_data):
        l, h, t_y = train_on_batch(X,p1,p2,y,diff)
        train_loss.append(l)
        train_mae.append(MAE(y,t_y))
        hing.append(h)
                     

        print('\rEpoch [%d/%d] Batch: %d%s' %(epoch + 1, epochs, batch, '.'*(batch%10)), end='')
    print('')
    print('Train loss:' +str(np.mean(train_loss)))
    print('Train MAE:' +str(np.mean(train_mae)))
    print('Hinge loss:' +str(np.mean(hing)))
    all_train_loss.append(np.mean(train_loss))
    all_train_mae.append(np.mean(train_mae))
        
    val_loss = []
    val_mae = []
                     
    for batch, (X,y,p1,p2,diff) in enumerate(test_data):
        v_loss, v_y = validate_on_batch(X,p1,p2,y,diff)
        val_loss.append(v_loss)
        val_mae.append(MAE(y,v_y))  
        
    
    if np.mean(val_mae) < best_val_mae:
        model.save_weights('./weights_mt_au_mae.h5')
        best_val_mae = np.mean(val_mae)
        
   
        
    print('Val loss: '+str(np.mean(val_loss)))    
    print('Val MAE: '+str(np.mean(val_mae))) 
    print('')
    
    
    all_val_loss.append(np.mean(val_loss))
    all_val_mae.append(np.mean(val_mae))

    if epoch > 3:       
        if (all_val_mae[epoch] + all_val_mae[epoch-1] + all_val_mae[epoch-2])/3 <  best_val_mae_MA:
            model.save_weights('./weights_mt_au_mae_MA.h5')
            best_val_mae_MA = (all_val_mae[epoch] + all_val_mae[epoch-1] + all_val_mae[epoch-2])/3
            
        
#Graph results
fig = plt.figure(figsize=(18,6))
x = list(range(1,len(all_train_loss)+1))
ax = fig.add_subplot(121)
ax.plot(x,all_train_loss, linewidth=1, label='train_loss', color = 'dodgerblue')
ax.plot(x,all_val_loss, linewidth=1, label='val_loss', color = 'orange')

ax.legend()
ax.set_xlabel('epoch')
ax.set_ylabel('loss')

                         
ax2 = fig.add_subplot(122)
                         
ax2.plot(x,all_train_mae, linewidth=1, label='train_mae', color = 'dodgerblue')                         
ax2.plot(x,all_val_mae, linewidth=1, label='val_mae', color = 'orange')

ax2.legend()
ax2.set_xlabel('epoch')
ax2.set_ylabel('mae')

