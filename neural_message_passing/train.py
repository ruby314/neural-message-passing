import os
import datetime
from functools import partial
import logging
from absl import app
from absl import flags
import tensorflow as tf
from tensorboard.plugins import projector

from message_passing import NeuralMessagePassing
from data import _extract_fn
from loss import contrastive_loss

# dataset specs
flags.DEFINE_string('dataset_name', 'mutag', 'name of dataset, in tfrecord file name')
flags.DEFINE_integer('num_atoms_vocab', 7, 'number of unique atoms in dataset')
flags.DEFINE_integer('num_edge_types', 12, 'number of unique bond types in dataset')
flags.DEFINE_integer('num_molecules', 188, 'number of molecules in dataset')
flags.DEFINE_integer('max_num_atoms', 28, 'maximum number of atoms in a molecule')
# model hyperparm
flags.DEFINE_integer('dim_h', 100, 'dimension of feature vectors')
flags.DEFINE_integer('time_steps', 3, 'number of time steps (levels) in message passing')
flags.DEFINE_integer('num_neg_samples', 10, 'number of negative samples (k)')
flags.DEFINE_float('gamma', 0.5, 'inverse temperature, as in paper')
# training
flags.DEFINE_integer('epochs', 2, 'training epochs')
flags.DEFINE_float('learning_rate', 0.1, 'learning rate')
# other
flags.DEFINE_string('logdir', '/tmp/tensorboard/logs', 'tensorboard log directory')
flags.DEFINE_bool('test_code', False, 'for testing code')
FLAGS = flags.FLAGS

LOGGING_FORMAT = '%(asctime)-15s %(message)s'
logging.basicConfig(format=LOGGING_FORMAT, level=logging.INFO)


def main (argv):

    # data
    dataset = tf.data.Dataset.list_files('./*{}*tfrecord'.format(FLAGS.dataset_name))
    dataset = dataset.interleave(tf.data.TFRecordDataset).map(_extract_fn)

    # message passing model
    nmp = NeuralMessagePassing(FLAGS.num_atoms_vocab, FLAGS.num_edge_types, FLAGS.dim_h)

    # molecule vectors u_m
    init_op = tf.keras.initializers.GlorotUniform(seed=None)
    init_op = partial(init_op, shape=(FLAGS.num_molecules, FLAGS.dim_h))
    u_m = tf.Variable(shape=(FLAGS.num_molecules, FLAGS.dim_h),
                           initial_value=init_op,
                           trainable=True,
                           dtype=tf.float32)

    # optimizer
    optimizer = tf.keras.optimizers.Adam(learning_rate=FLAGS.learning_rate)

    # take small subset of data if testing code
    if FLAGS.test_code:
        num_molecules_testing = 5
        FLAGS.num_neg_samples = 2
        dataset = dataset.take(num_molecules_testing)
    num_molecules_used = num_molecules_testing if FLAGS.test_code else FLAGS.num_molecules

    # tensorboard
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%M%H%S')
    train_log_dir = os.path.join(FLAGS.logdir, timestamp)
    summary_writer = tf.summary.create_file_writer(train_log_dir)
    loss_metric = tf.keras.metrics.Mean('train/loss', dtype=tf.float32)
    logging.info('logging to {}'.format(train_log_dir))

    # tensorboard projector setup for visualizing feature vectors
    projector_config = projector.ProjectorConfig()
    embedding = projector_config.embeddings.add()
    embedding.tensor_name = 'embedding/.ATTRIBUTES/VARIABLE_VALUE'
    embedding.metadata_path = 'metadata.tsv'
    # save labels to metadata file
    with open(os.path.join(train_log_dir, 'metadata.tsv'), 'w') as f:
        for _, _, label, _ in dataset:
            f.write('{}\n'.format(label))

    logging.info('start training')

    # unsupervised training
    for epoch in range(FLAGS.epochs):

        # compute h_v for all graphs (molecules)
        h_v_all_molecules = tf.TensorArray(tf.float32, size=0, dynamic_size=True, clear_after_read=True)
        with tf.GradientTape() as tape:

            num_atoms = []  # in each molecule
            for m, (atoms, adjs, label, graphid) in enumerate(dataset):

                num_atoms.append(len(atoms))

                # message passing
                h_v = nmp(atoms, adjs, FLAGS.time_steps)  # all levels for current molecule

                # zero pad h_v so dimension is same for all molecules, to write to TensorArray
                zero_pad = tf.zeros((FLAGS.time_steps, FLAGS.max_num_atoms - len(atoms), FLAGS.dim_h), dtype=tf.float32)
                h_v_padded = tf.concat([h_v, zero_pad], axis=1)
                h_v_all_molecules.write(m, h_v_padded)

            h_v_all_molecules = h_v_all_molecules.stack()  # (num_molecules, timesteps, max_num_atoms, dim_h)
            logging.info('computed all molecule representations in tensor of shape {}'.format(h_v_all_molecules.shape))

            # objective
            loss = contrastive_loss(h_v_all_molecules, u_m, num_atoms, FLAGS.num_neg_samples, FLAGS.gamma)

        # backprop
        trainable_variables = nmp.trainable_variables + [u_m]
        gradients = tape.gradient(loss, trainable_variables)
        optimizer.apply_gradients(zip(gradients, trainable_variables))
        logging.info('epoch {}, loss {}'.format(epoch, loss[0]))
        loss_metric(loss)

        # tensorboard
        with summary_writer.as_default():
            tf.summary.scalar('train/loss', loss_metric.result(), step=epoch)
        loss_metric.reset_states()

        # embeddings viz
        features = []
        for m, h_v in enumerate(h_v_all_molecules):
            # summary is the readout feature that would be used in downstream task,
            # except unlike paper no projection W, which is learned via supervision
            summary = tf.zeros((1, FLAGS.dim_h), dtype=tf.float32)
            for ell in range(FLAGS.time_steps):
                for v in range(num_atoms[m]):
                    summary += tf.nn.softmax(h_v[ell, v, :])
            features.append(summary)
        features = tf.Variable(tf.concat(features, axis=0))
        checkpoint = tf.train.Checkpoint(embedding=features)
        checkpoint.save(os.path.join(train_log_dir, 'embedding.ckpt'))
        projector.visualize_embeddings(train_log_dir, projector_config)


if __name__ == '__main__':
    app.run(main)
