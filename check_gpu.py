import sys
# import pandas as pd
# import sklearn as sk
from tensorflow import keras
# from keras.api._v2.keras import layers
import tensorflow as tf
import platform

# NOTE:run this file after activating tf-gpu environment
print(platform.platform())
print(f"Tensor Flow Version: {tf.__version__}")
# print(f"Keras Version: {keras.__version__}")
print(f"Python {sys.version}")
# # print(f"Pandas {pd.__version__}")
# # print(f"Scikit-Learn {sk.__version__}")
gpu = len(tf.config.list_physical_devices('GPU')) > 0
print("GPU is", "available" if gpu else "NOT AVAILABLE")
