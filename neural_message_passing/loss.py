import tensorflow as tf
import numpy as np


def contrastive_loss(h_v_all_molecules, u_m, num_atoms_in_molecules, num_neg_samples, gamma):
    """contrastive unsupervised loss"""

    num_molecules, time_steps, _, _ = h_v_all_molecules.shape
    molecule_indices = list(range(num_molecules))

    loss = tf.zeros((1, ), dtype=tf.float32)
    for m, h_v in enumerate(h_v_all_molecules):
        for ell in range(time_steps):
            for v in range(num_atoms_in_molecules[m]):

                # positive
                h_pos = h_v[ell, v, :]
                logit = tf.tensordot(u_m[m, :], h_pos, axes=1)  # dot product
                loss -= tf.math.log(tf.nn.sigmoid(gamma * logit))

                # negative samples
                candidates = molecule_indices[:m] + molecule_indices[m+1:]
                m_neg = np.random.choice(candidates, num_neg_samples, replace=False)
                for m_n in m_neg:

                    # randomly sample atom from this molecule
                    v_prime = np.random.randint(num_atoms_in_molecules[m_n])
                    # candidate negative feature
                    h_neg = h_v_all_molecules[m_n, ell, v_prime, :]

                    # resample if randomly sampled negative matches positive feature
                    # implementation below a little hacky, but works
                    accepted = False
                    while not accepted:
                        try:
                            tf.debugging.assert_near(h_pos, h_neg)
                            # h_neg too close to h_pos, draw new h_neg
                            # sample new molecule - may overlap other sampled neg molecules
                            m_n = np.random.choice(candidates, 1)[0]
                            v_prime = np.random.randint(num_atoms_in_molecules[m_n])
                            h_neg = h_v_all_molecules[m_n, ell, v_prime, :]
                        except tf.errors.InvalidArgumentError:
                            # h_pos and h_neg are different, accept h_neg
                            accepted = True

                    logit = tf.tensordot(u_m[m, :], h_neg, axes=1)  # dot product
                    loss -= tf.math.log(tf.nn.sigmoid( - gamma * logit))

    loss = loss / (num_molecules * time_steps)

    return loss
