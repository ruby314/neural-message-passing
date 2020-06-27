import tensorflow as tf
import numpy as np

# from https://github.com/pfnet-research/hierarchical-molecular-learning
# also requires MUTAG.mat from that repo
from load_mutag import load_whole_data


def _int64_feature(scalar):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[scalar]))


def _bytes_feature_from_arr(value):
    serialized_value = tf.io.serialize_tensor(value).numpy()
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[serialized_value]))


def create_training_example(graphid, atom_arr, adjs, label):

    features = {
        'atom_array': _bytes_feature_from_arr(atom_arr),
        'adjacency': _bytes_feature_from_arr(adjs),
        'label': _int64_feature(label),
        'graphid': _int64_feature(graphid)
    }

    return tf.train.Example(features=tf.train.Features(feature=features))

def _extract_fn(record):

    example_fmt = {
        'atom_array': tf.io.FixedLenFeature([], tf.string),
        'adjacency': tf.io.FixedLenFeature([], tf.string),
        'label': tf.io.FixedLenFeature([], tf.int64),
        'graphid': tf.io.FixedLenFeature([], tf.int64),
    }

    parsed_example = tf.io.parse_single_example(record, example_fmt)
    atom_array = tf.io.parse_tensor(parsed_example['atom_array'], out_type=tf.int32)
    adj = tf.io.parse_tensor(parsed_example['adjacency'], out_type=tf.int32)
    
    return atom_array, adj, parsed_example['label'], parsed_example['graphid']


if __name__ == '__main__':

    results = load_whole_data('')  # function ignores input filename

    savepath = './mutag.tfrecord'
    
    with tf.io.TFRecordWriter(savepath) as writer:    

        for graphid, atom_arr, adjs, label in results:
            adjs = adjs.astype(np.int32)

            # strip zero pad from atom_arr
            num_atoms = np.sum(atom_arr != 0)
            atom_arr = atom_arr[:num_atoms]

            example_proto = create_training_example(graphid, atom_arr, adjs, label)
            writer.write(example_proto.SerializeToString())
