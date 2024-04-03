
import tensorflow as tf
print("TensorFlow version: ", tf.__version__)
import tensorflow as tf
print("Is TensorFlow built with CUDA: ", tf.test.is_built_with_cuda())
print("CUDA version: ", tf.sysconfig.get_build_info()['cuda_version'])
print("cuDNN version: ", tf.sysconfig.get_build_info()['cudnn_version'])
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
