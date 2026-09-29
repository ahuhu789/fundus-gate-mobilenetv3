import os, sys
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import tensorflow as tf

keras_path = '/kaggle/working/fundus_gate.keras'
tflite_path = '/kaggle/working/fundus_gate.tflite'
tflite_fp16_path = '/kaggle/working/fundus_gate_float16.tflite'

print('Loading Keras model on CPU for clean TFLite conversion...')
model = tf.keras.models.load_model(keras_path)
run_model = tf.function(lambda x: model(x))
concrete_func = run_model.get_concrete_function(tf.TensorSpec([1, 224, 224, 3], tf.float32))

print('Converting to Float32 TFLite...')
converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func])
tflite_model = converter.convert()
with open(tflite_path, 'wb') as f:
    f.write(tflite_model)
print(f'Saved TFLite model: {tflite_path} ({os.path.getsize(tflite_path)/1024/1024:.2f} MB)')

print('Converting to Float16 Quantized TFLite...')
converter_fp16 = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func])
converter_fp16.optimizations = [tf.lite.Optimize.DEFAULT]
converter_fp16.target_spec.supported_types = [tf.float16]
tflite_fp16_model = converter_fp16.convert()
with open(tflite_fp16_path, 'wb') as f:
    f.write(tflite_fp16_model)
print(f'Saved TFLite Float16 model: {tflite_fp16_path} ({os.path.getsize(tflite_fp16_path)/1024/1024:.2f} MB)')
print('TFLite CPU Conversion Complete!')