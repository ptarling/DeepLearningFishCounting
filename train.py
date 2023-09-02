import numpy as np
import tensorflow as tf
from model import *
from lossfunctions_metrics import *
import matplotlib.pyplot as plt
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
"""
第二个部分代码涉及到平均移动平均（Moving Average，MA）的概念，它可能被用于更加稳定地跟踪验证集性能的变化。
具体来说，这段代码在训练的后期（epoch > 3）使用了连续三个 epoch 的验证集 MAE 的平均值。如果这个平均值小于之前的最佳验证集 MAE（best_val_mae_MA），那么它会保存当前模型的权重，并更新 best_val_mae_MA 为这个平均值。这种方法可以降低随机波动对判断模型性能的影响，因为取三个连续 epoch 的平均值可能会减少噪声。
总的来说，第二个部分的代码可能是为了稳定地保存在连续几个 epoch 内性能有持续提升的模型权重，而不仅仅考虑单个 epoch 的性能。这种方式有助于避免因为单个 epoch 变化的随机性而导致的过早保存模型。
"""

# Note:这个代码没有对density map的shape进行处理，因为model会将输入数据缩小维度的，但是代码却仍然可以运行，因为算loss的是基于sum图像，因此x和yshape不一样也能算loss，但是not make sense

# Load data
x_train = np.load('./data/Labelled_data/x_train.npy')
y_train = np.load('./data/Labelled_data/y_train.npy')


ind_shuff = np.random.permutation(len(x_train))
x_train = x_train[ind_shuff]
y_train = y_train[ind_shuff]

x_val = np.load('./data/Labelled_data/x_val.npy')
y_val = np.load('./data/Labelled_data/y_val.npy')

pair1_train = np.load('./data/unlabelled_data/pair1_train.npy')
pair2_train = np.load('./data/unlabelled_data/pair2_train.npy')

pair1_val = np.load('./data/unlabelled_data/pair1_val.npy')
pair2_val = np.load('./data/unlabelled_data/pair2_val.npy')


rank_train = np.zeros(len(pair1_train))
rank_val = np.zeros(len(pair1_val))


# change shape for variance prediction:
def y_reshape(y_data):
    y_data = np.expand_dims(y_data, -1)
    y_data_reshape = np.concatenate((y_data, np.zeros((y_data.shape))), axis=3)
    return y_data_reshape


# ground true变成一个channel为2的tensor，目的应该是ground true需要density map和noise variance
y_train_reshape = y_reshape(y_train)
y_val_reshape = y_reshape(y_val)


# Convert all data to tensors
x_train = tf.convert_to_tensor(x_train, dtype=tf.float32)
y_train_reshape = tf.convert_to_tensor(y_train_reshape, dtype=tf.float32)
x_val = tf.convert_to_tensor(x_val, dtype=tf.float32)
y_val_reshape = tf.convert_to_tensor(y_val_reshape, dtype=tf.float32)
pair1_train = tf.convert_to_tensor(pair1_train, dtype=tf.float32)
pair2_train = tf.convert_to_tensor(pair2_train, dtype=tf.float32)
pair1_val = tf.convert_to_tensor(pair1_val, dtype=tf.float32)
pair2_val = tf.convert_to_tensor(pair2_val, dtype=tf.float32)
rank_train = tf.convert_to_tensor(rank_train, dtype=tf.float32)
rank_val = tf.convert_to_tensor(rank_val, dtype=tf.float32)


# Build model
model = multitask_au(input_shape=(576, 320, 3))
# option to load weights from our trained model
# model.load_weights('./weights_mt_au_mae.h5')

optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)


@tf.function
# diff就是一个以batchsize为形状的0
def train_on_batch(X, p1, p2, y, diff):
    with tf.GradientTape() as tape:
        yhat, yhat_diff = model([X, p1, p2], training=True)
        yhat = tf.convert_to_tensor(yhat, dtype=tf.float32)
        yhat_diff = tf.convert_to_tensor(yhat_diff, dtype=tf.float32)

        ab = au_absolute_loss(y, yhat)
        hinge = hinge_loss(diff, yhat_diff)
        # loss为单任务的loss和ranking loss相加
        loss_value = tf.reduce_sum(ab + hinge)

    grads = tape.gradient(loss_value, model.trainable_weights)
    optimizer.apply_gradients(zip(grads, model.trainable_weights))
    return loss_value, hinge, yhat


@tf.function
def validate_on_batch(X, p1, p2, y, diff):
    yhat_val, yhat_diff_val = model([X, p1, p2], training=False)
    yhat_val = tf.convert_to_tensor(yhat_val, dtype=tf.float32)
    yhat_diff_val = tf.convert_to_tensor(yhat_diff_val, dtype=tf.float32)
    ab = au_absolute_loss(y, yhat_val)
    hinge = hinge_loss(diff, yhat_diff_val)

    loss_value_val = tf.reduce_sum(ab + hinge)
    return loss_value_val, yhat_val


batch_size = 4
epochs = 5


best_val_mae = 99999
best_val_mae_MA = 99999

all_train_loss = []
all_train_mae = []
all_val_loss = []
all_val_mae = []


for epoch in range(0, epochs):
    # 这3个列表存放的是，每个batch的loss，mae，和hinge
    train_loss = []
    train_mae = []
    hing = []
    # tf.data.Dataset.from_tensor_slices 类似torch中的dataset，可以将数据进行包裹后，然后获取时就可以获取batch size大小
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

    for batch, (X, y, p1, p2, diff) in enumerate(train_data):
        l, h, t_y = train_on_batch(X, p1, p2, y, diff)
        train_loss.append(l)
        train_mae.append(MAE(y, t_y))
        hing.append(h)

        print('\rEpoch [%d/%d] Batch: %d%s' %
              (epoch + 1, epochs, batch, '.' * (batch % 10)), end='')
    print('')
    print('Train loss:' + str(np.mean(train_loss)))
    print('Train MAE:' + str(np.mean(train_mae)))
    print('Hinge loss:' + str(np.mean(hing)))
    all_train_loss.append(np.mean(train_loss))
    all_train_mae.append(np.mean(train_mae))

    # 存储validation里每个batch的loss和mae
    val_loss = []
    val_mae = []

    # 每个epoch，train和validation都跑一遍
    for batch, (X, y, p1, p2, diff) in enumerate(test_data):
        v_loss, v_y = validate_on_batch(X, p1, p2, y, diff)
        val_loss.append(v_loss)
        val_mae.append(MAE(y, v_y))

    # 同样，每个epoch都会判断最好的mae
    if np.mean(val_mae) < best_val_mae:
        model.save_weights('./weights_mt_au_mae.h5')
        best_val_mae = np.mean(val_mae)

    print('Val loss: ' + str(np.mean(val_loss)))
    print('Val MAE: ' + str(np.mean(val_mae)))
    print('')

    all_val_loss.append(np.mean(val_loss))
    all_val_mae.append(np.mean(val_mae))

    if epoch > 3:
        if (all_val_mae[epoch] + all_val_mae[epoch - 1] + all_val_mae[epoch - 2]) / 3 < best_val_mae_MA:
            model.save_weights('./weights_mt_au_mae_MA.h5')
            best_val_mae_MA = (
                all_val_mae[epoch] + all_val_mae[epoch - 1] + all_val_mae[epoch - 2]) / 3


# Graph results
fig = plt.figure(figsize=(18, 6))
x = list(range(1, len(all_train_loss) + 1))
ax = fig.add_subplot(121)
ax.plot(x, all_train_loss, linewidth=1, label='train_loss', color='dodgerblue')
ax.plot(x, all_val_loss, linewidth=1, label='val_loss', color='orange')

ax.legend()
ax.set_xlabel('epoch')
ax.set_ylabel('loss')


ax2 = fig.add_subplot(122)

ax2.plot(x, all_train_mae, linewidth=1, label='train_mae', color='dodgerblue')
ax2.plot(x, all_val_mae, linewidth=1, label='val_mae', color='orange')

ax2.legend()
ax2.set_xlabel('epoch')
ax2.set_ylabel('mae')
