from object_detection.protos import preprocessor_pb2
import tensorflow as tf
dataset = tf.data.Dataset.from_tensor_slices("pipeline.config")

decoder = tf_example_decoder.TFExampleDecoder(
    label_map_proto_file= "preprocessor.proto",
    use_display_name=False,
    preprocessing_options=preprocessor_pb2.PreprocessingOptions())

dataset = dataset.map(lambda x: decoder.decode(x))