from tfcoreml.convert import convert

coreml_model = convert(tf_model_path='your_tf_model_directory/saved_model', mlmodel_path='output.mlmodel')
 
