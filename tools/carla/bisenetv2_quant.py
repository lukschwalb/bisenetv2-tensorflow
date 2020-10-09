# https://github.com/PINTO0309/PINTO_model_zoo

import tensorflow as tf
import shutil
import os
import argparse
from tensorflow.python.saved_model import tag_constants
import numpy as np

export_dir = 'carla_bisenet_fullint'
BATCH_SIZE = 16

parser = argparse.ArgumentParser(description='Quantize BiSeNetv2 Tensorflow Graph')
parser.add_argument('graph', help="Graph definition")
parser.add_argument('--width', type=int, default=255)
parser.add_argument('--height', type=int, default=255)
parser.add_argument('--test-data', default='carla_test_data.npy')

args        = parser.parse_args()
graph_file  = args.graph
width       = args.width
height      = args.height
test_data   = args.test_data

raw_test_data = np.load(test_data, allow_pickle=True)

def representative_dataset_gen():
    for image in raw_test_data:
        image = tf.image.resize(image, (height, width))
        image = image[np.newaxis,:,:,:]
        image = image - 127.5
        image = image * 0.007843
        yield [image]

def get_graph_def_from_file():
    tf.compat.v1.reset_default_graph()
    with tf.compat.v1.Graph().as_default():
        with tf.compat.v1.gfile.GFile(graph_file, 'rb') as f:
            graph_def = tf.compat.v1.GraphDef()
            graph_def.ParseFromString(f.read())
            return graph_def

def convert_graph_def_to_saved_model(input_name):
    outputs = ['BiseNetV2/prob:0']
    graph_def = get_graph_def_from_file()
    with tf.compat.v1.Session(graph=tf.Graph()) as session:
        tf.import_graph_def(graph_def, name='')
        for node in graph_def.node:
            if node.op=='Placeholder':
                print(session.graph.get_tensor_by_name('{}:0'.format(node.name)).shape)

        tf.compat.v1.saved_model.simple_save(
            session,
            export_dir,
            inputs={input_name: session.graph.get_tensor_by_name('{}:0'.format(node.name))
                for node in graph_def.node if node.op=='Placeholder'},
            outputs={t.rstrip(":0"):session.graph.get_tensor_by_name(t) for t in outputs}
        )
        print('Optimized graph converted to SavedModel!')


shutil.rmtree(export_dir, ignore_errors=True)
convert_graph_def_to_saved_model(graph_file)

# Full Integer Quantization - Input/Output=float32
converter = tf.lite.TFLiteConverter.from_saved_model(export_dir)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8,tf.lite.OpsSet.SELECT_TF_OPS]
converter.inference_input_type = tf.uint8
converter.inference_output_type = tf.uint8
converter.representative_dataset = representative_dataset_gen
tflite_model = converter.convert()
with open('bisenetv2_carla_{}x{}_full_integer_quant.tflite'.format(height, width), 'wb') as w:
    w.write(tflite_model)
    print('Full Integer Quantization complete! - bisenetv2_carla_{}x{}_full_integer_quant.tflite'.format(height, width))
