import io
import logging
import cv2
import numpy as np

# import tensorflow as tf
import tensorflow as tf
from object_detection.inputs import train_input
from object_detection.protos import input_reader_pb2
from object_detection.builders.dataset_builder import build as build_dataset
from object_detection.utils.config_util import get_configs_from_pipeline_file
from tools.waymo_reader.simple_waymo_open_dataset_reader import WaymoDataFileReader, dataset_pb2
from PIL import Image
import glob
resize_ratio = 0.5
def getGroundTruth(frame):
  labelMap = {
    "TYPE_VEHICLE" : 1,
    "TYPE_PEDESTRIAN" : 2,
    "TYPE_CYCLELIST" : 4,
  }
  camera_name = dataset_pb2.CameraName.FRONT
  labelsFrontCamera = [frame.labels for frame in frame if frame.name == camera_name][0]
  labelsBox = [{
    "center_x" : label.box.center_x * resize_ratio,
    "center_y" : label.box.center_y * resize_ratio,
    "width" : label.box.width * resize_ratio,
    "length" : label.box.length * resize_ratio,
    } for label in labelsFrontCamera]
  labelsClass = [label.type for label in labelsFrontCamera]
  return labelsBox, labelsClass
  
def getImage(frame):

    # load the camera data structure
    camera_name = dataset_pb2.CameraName.FRONT
    image = [obj for obj in frame.images if obj.name == camera_name][0]

    # convert the actual image into rgb format
    img = np.array(Image.open(io.BytesIO(image.image)))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # resize the image to better fit the screen
    dim = (int(img.shape[1] * resize_ratio), int(img.shape[0] * resize_ratio))
    resized = cv2.resize(img, dim)

    # display the image 
    # cv2.imshow("Front-camera image", resized)
    # cv2.waitKey(0)
    return resized
    
    
def get_dataset(tfrecord_path, label_map='label_map.pbtxt'):
    """
    Opens a tf record file and create tf dataset
    args:
      - tfrecord_path [str]: path to a tf record file
      - label_map [str]: path the label_map file
    returns:
      - dataset [tf.Dataset]: tensorflow dataset
    """
      
    input_config = input_reader_pb2.InputReader()
    input_config.label_map_path = label_map
    input_config.tf_record_input_reader.input_path[:] = [tfrecord_path]
    
    dataset = build_dataset(input_config)
    return dataset
    
    
    dataset = []
    tfrecordList = list(glob.glob("data/train/*.tfrecord"))
    for tfrecord_path in tfrecordList:
      dataset.append(WaymoDataFileReader(tfrecord_path))
    # dataset.append(WaymoDataFileReader(tfrecord_path))
    suffled_frame = []
    for data in dataset:
      for frame in data:
        suffled_frame.append((frame, frame.camera_labels))
    import random
    random.shuffle(suffled_frame)
    # dataset_iter = iter(dataset)
    # for idx, frame in enumerate(dataset_iter):
    #   display_image(frame)



    return suffled_frame


def get_module_logger(mod_name):
    """ simple logger """
    logger = logging.getLogger(mod_name)
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s %(levelname)-8s %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)
    return logger


def get_train_input(config_path):
  """
  Get the tf dataset that inputs training batches
  args:
    - config_path [str]: path to the edited config file
  returns:
    - dataset [tf.Dataset]: data outputting augmented batches
  """
  # parse config
  configs = get_configs_from_pipeline_file(config_path)
  train_config = configs['train_config']
  train_input_config = configs['train_input_config']

  # get the dataset
  dataset = train_input(train_config, train_input_config, configs['model'])
  return dataset

def parse_frame(frame, camera_name='FRONT'):
    """ 
    take a frame, output the bboxes and the image

    dataset = tf.data.TFRecordDataset(FILENAME, compression_type='')
      for data in dataset:
      frame = open_dataset.Frame()
      frame.ParseFromString(bytearray(data.numpy()))
    
    args:
      - frame [waymo_open_dataset.dataset_pb2.Frame]: a waymo frame, contains images and annotations
      - camera_name [str]: one frame contains images and annotations for multiple cameras
    
    returns:
      - encoded_jpeg [bytes]: jpeg encoded image
      - annotations [protobuf object]: bboxes and classes
    """
    # get image
    images = frame.images
    for im in images:
        if open_dataset.CameraName.Name.Name(im.name) != camera_name:
            continue
        encoded_jpeg = im.image
    
    # get bboxes
    labels = frame.camera_labels
    for lab in labels:
        if open_dataset.CameraName.Name.Name(lab.name) != camera_name:
            continue
        annotations = lab.labels
    return encoded_jpeg, annotations


def int64_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def int64_list_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def bytes_list_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))


def float_list_feature(value):
  return tf.train.Feature(float_list=tf.train.FloatList(value=value))