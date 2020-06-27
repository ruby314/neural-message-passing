import tensorflow as tf
from functools import partial


class NeuralMessagePassing(tf.keras.Model):

    def __init__(self, num_atoms_vocab, num_edge_types, dim_h, **kwargs):
        """message passing"""

        super(NeuralMessagePassing, self).__init__(**kwargs)

        self.dim_h = dim_h
        self.num_edge_types = num_edge_types

        # atom embedding
        self.embedding = tf.keras.layers.Embedding(num_atoms_vocab+1, dim_h)  # account for null atom 0

        # H_edges contains weight matrices for each edge type
        init_op = tf.keras.initializers.GlorotUniform(seed=None)
        init_op = partial(init_op, shape=(num_edge_types, dim_h, dim_h))
        self.H_edges = tf.Variable(shape=(num_edge_types, dim_h, dim_h),
                                   initial_value=init_op,
                                   trainable=True,
                                   dtype=tf.float32)

    def call(self, atom_arr, adjs, max_time_steps):
        """
        returns feature representations for all nodes at all levels:
        returned tensor is max_time_steps x max num atoms x self.dim_h
        for single molecule, extend to batches later
        """

        embedded = self.embedding(atom_arr)  # num_atoms x dim_h

        # initialize to atom embedding
        # in related literature, h_v_0 may be initialized differently?
        h_v_0 = embedded

        h_v = h_v_0
        h_v_all = tf.TensorArray(tf.float32, size=0, dynamic_size=True, clear_after_read=True)
        for t in range(max_time_steps):

            # construct message for every node (atom) in molecule
            message_list = []
            num_atoms = len(atom_arr)
            for v in range(len(atom_arr)):
                m_v_t = self.get_message(v, adjs, h_v, num_atoms)
                message_list.append(m_v_t)

            # convert message list to tensor, zero padded
            messages = tf.concat(message_list, axis=0)

            # update
            h_v = self.update(h_v, messages)

            # save feature vector at this level
            h_v_all.write(t, h_v)

        # convert to tensor
        features = h_v_all.stack()

        return features

    def get_message(self, atom_index, adjs, h_v, num_atoms):

        dim_h = self.dim_h

        m_v_t = tf.zeros((dim_h,), dtype=tf.float32)
        # contribution from every edge type
        for edge_index in range(self.num_edge_types):

            bonds = adjs[edge_index, atom_index, :num_atoms]
            bonds = tf.expand_dims(bonds == 1, axis=0)  # convert to bool tensor of dim 1 x num_atoms

            selector = tf.tile(bonds, [dim_h, 1])  # dim_h x num_atoms
            h_v_transpose = tf.transpose(h_v)
            inp = tf.where(selector, h_v_transpose, tf.zeros(h_v_transpose.shape, dtype=tf.float32))

            m_v_t_edge = tf.reduce_sum(tf.linalg.matmul(self.H_edges[edge_index, :, :], inp), axis=-1)
            m_v_t += m_v_t_edge

        m_v_t = tf.expand_dims(m_v_t, axis=0)

        return m_v_t

    def update(self, h_v, messages):

        h_v_updated = tf.nn.sigmoid(h_v + messages)

        return h_v_updated
