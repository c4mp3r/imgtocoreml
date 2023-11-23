import coremltools
import numpy as np
from PIL import Image

# Load the CoreML model
coreml_model = coremltools.models.MLModel('image_translation_model.mlmodel')

# Prepare input data (adjust this for your specific model)
# For example, if your model takes RGB images of size 256x256 as input:
input_data = np.array(Image.open('input_image.jpg').resize((256, 256)))
input_data = input_data.astype(np.float32) / 255.0  # Normalize the input data

# Make predictions
input_features = {'input': input_data}
predictions = coreml_model.predict(input_features)

# Process the output (adjust this for your specific model)
output_data = predictions['output']

# You may need to perform additional post-processing depending on your model's output format

# If your model produces color images, you can save the output as an image
output_image = Image.fromarray((output_data * 255).astype(np.uint8))
output_image.save('output_image.jpg')
 
