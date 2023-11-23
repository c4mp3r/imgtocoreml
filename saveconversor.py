from tfcoreml.convert import convert
import tensorflow as tf

# Load the trained TensorFlow model
tensorflow_model = tf.keras.models.load_model('image_translation_generator.h5')

# Convert the TensorFlow model to CoreML
coreml_model = convert(tf_model_path=tensorflow_model, input_name='input', output_name='output')

# Save the CoreML model
coreml_model.save('image_translation_model.mlmodel')
 
