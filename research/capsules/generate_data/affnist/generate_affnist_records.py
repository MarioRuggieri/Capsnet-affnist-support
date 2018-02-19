import os
import numpy as np
import scipy.io as sio
import tensorflow as tf
from itertools import izip

FLAGS = tf.flags.FLAGS

tf.flags.DEFINE_string('data_dir', '../../data/affnist_data', 'The affnist data directory.')
tf.flags.DEFINE_string('testdata_dir', '../../testdata/affnist/', 'The affnist datatest directory')

# image size
IMG_LEN = 40; IMG_DEPTH = 1

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def read_samples(path):
    struct = sio.loadmat(path)
    mat = struct['affNISTdata']
    images = mat['image'][0][0].transpose()
    labels = mat['label_int'][0][0].squeeze()
    return images,labels

def read_data(path, type):
    if type == 'train':
        listdir = os.listdir(path)
        (images,labels) = read_samples(os.path.join(path, listdir[0]))
        for batch_file in listdir[1:]:
            batch_images, batch_labels = read_samples(os.path.join(path, batch_file))
            images = np.concatenate((images,batch_images))
            labels = np.concatenate((labels,batch_labels))
    else:
        images,labels = read_samples(path)
    return images,labels

def write_example(image, label, writer):
    # create feature
    feature = {'image_raw': _bytes_feature(image.tostring()),
               'label': _int64_feature(label),
               'height': _int64_feature(IMG_LEN),
               'width': _int64_feature(IMG_LEN),
               'depth': _int64_feature(IMG_DEPTH)}
    # create an example protocol buffer
    example = tf.train.Example(features=tf.train.Features(feature=feature))
    # write the serialized example
    writer.write(example.SerializeToString())

def save_tfrecords(images,labels,filename,testfilename):
    #generate tfrecords for affnist_input_record_test
    writer = tf.python_io.TFRecordWriter(testfilename)
    write_example(images[0],labels[0],writer)
    writer.close()
    #generate tfrecords for affnist_input_record
    writer = tf.python_io.TFRecordWriter(filename)
    for image,label in izip(images,labels):
        write_example(image,label,writer)
    writer.close()

if __name__ == "__main__":
    train_records_path = os.path.join(FLAGS.data_dir,'affnist_train.tfrecords')
    train_datatest_path = os.path.join(FLAGS.testdata_dir,'affnist_train.tfrecords')
    test_records_path = os.path.join(FLAGS.data_dir,'affnist_test.tfrecords')
    test_datatest_path = os.path.join(FLAGS.testdata_dir, 'affnist_test.tfrecords')

    if not os.path.exists(train_records_path):
        print("Loading training data...")
        train_img, train_lab = read_data(os.path.join(FLAGS.data_dir,'training_batches'),'train')
        print("Number of training samples: " + str(train_img.shape[0]) + "\nGenerating tfrecords file...")
        save_tfrecords(train_img, train_lab, train_records_path, train_datatest_path)
    else:
        print("Training records already generated!")

    if not os.path.exists(test_records_path):
        print("Loading testing data...")
        test_img, test_lab = read_data(os.path.join(FLAGS.data_dir,'test.mat'),'test')
        print("Number of test samples: " + str(test_img.shape[0]) + "\nGenerating tfrecords file...")
        save_tfrecords(test_img, test_lab, test_records_path, test_datatest_path)
    else:
        print("Testing records already generated!")

    print("finished!")