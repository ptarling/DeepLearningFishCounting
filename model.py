import tensorflow as tf


def multitask_au (input_shape = (576,320,3)):
    RESNET = tf.keras.applications.ResNet50(weights='imagenet', include_top=False) #can intialise with ImageNet weights
    mod = tf.keras.Sequential(RESNET)
    mod.add(tf.keras.layers.Conv2D(2, (1, 1), activation = 'relu'))
    
    input_shape = (input_shape)
    
    #inputs
    input_count = tf.keras.Input(shape=input_shape)
    input_pair1 = tf.keras.Input(shape=input_shape)
    input_pair2 = tf.keras.Input(shape=input_shape)
    
    #outputs
    output_count = mod(input_count)
    output_pair1 = mod(input_pair1)
    output_pair2 = mod(input_pair2)
    
    #add global average pooling layer to each branch of Siamese network and subtract pair1 from pair2
    output_pair1 = tf.keras.layers.GlobalAveragePooling2D()(output_pair1)
    output_pair2 = tf.keras.layers.GlobalAveragePooling2D()(output_pair2)
    pair_diff = tf.keras.layers.Subtract()([output_pair2, output_pair1])
    
    model = tf.keras.Model(inputs=[input_count,input_pair1, input_pair2], outputs=[output_count, pair_diff])
    
    return model



