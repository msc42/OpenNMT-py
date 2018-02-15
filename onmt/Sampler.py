from __future__ import division

import numpy as np
import scipy.misc as misc
import torch


class Sampler(object):

    def __init__(self):
        pass

    def sample(self, tgt_data):
        """Return augmented tgt_data, rewards, and weights of proposed distribution"""
        pass

    def reward(self, y, y_ast, score=None):
        """Return reward of (y, y*)"""
        pass


class HammingDistanceSampler(Sampler):

    def __init__(self, temperature, max_len, voc_min, voc_max):
        """Sampling augmented target sentences and importance weights.
        EXAMPLE
        -------
        >>> edit_sampler = HammingDistanceSampler(0.5, 10, 0, 9)
        >>> tgt_data = [torch.LongTensor([0, 3, 5, 4, 3]), torch.LongTensor([2, 5, 1, 3, 9]), torch.LongTensor([2, 0, 1, 4, 5])]
        >>> augmented_tgt_data, rewards, proposed_weights = edit_sampler.sample(tgt_data, [2, 3, 4])
        >>> for s in tgt_data:
        ...     print(s)
        >>> for s in augmented_tgt_data:
        ...     print(s)
        >>> for r in rewards:
        ...     print(r)
        >>> for p in proposed_weights:
        ...     print(p)
        ARGUMENTS
        ---------
        voc_min: minimum index of vocabulary (inclusive)
        voc_max: maximum index of vocabulary (inclusive)
        """
        super(HammingDistanceSampler, self).__init__()
        self.temperature = temperature
        self.max_len = max_len
        self.voc_min = voc_min
        self.voc_max = voc_max
        self.voc_size = voc_max - voc_min + 1
        self.edit_frac = 0.2

        p = np.ones((self.max_len + 1, self.max_len))  # p[len_target, n_edit] = values of proposal distributions
        self.p = p

    def sample(self, tgt_data, edit_list=None):
        """Augment tgt_data and return importance sampling weight
        ARGS
        ----
        tgt_data: list of Torch.LongTensor
        edit_list: specify the edit length for each sample (especially for debug)
        Types of augmentation include: - deletion
        """

        if edit_list is not None:
            assert len(edit_list) == len(tgt_data)

        new_tgt_data = []
        rewards = []
        proposal_weights = []
        for i, tgt in enumerate(tgt_data):
            len_tgt = len(tgt)
            max_edit = int(len_tgt * self.edit_frac)  # in this study, we define max_edit as length of sentence

            # define edit distance for this tgt sentence
            if edit_list is not None:
                e = edit_list[i]
            else:
                e = np.random.choice(max_edit + 1, 1)[0]  # choose from {0, ..., max_edit}

            # get proposal weights
            proposal_weights.append(self.p[len_tgt, e])

            # execute augmentation
            n_substitutions = e
            # substitutions
            substituted_ixs = list(np.random.choice(len_tgt, n_substitutions, replace=False))
            new_tgt = []
            for j, t in enumerate(tgt):
                if j in substituted_ixs:
                    new_token = t
                    while new_token == t:
                        new_token = int(np.random.choice(self.voc_size, 1)[0]) + self.voc_min
                    new_tgt.append(new_token)
                else:
                    new_tgt.append(t)

            # store the corresponding reward
            reward = float(self.reward(tgt, new_tgt, e))
            rewards.append(reward)

            new_tgt_data.append(torch.LongTensor(new_tgt))

        return new_tgt_data, rewards, proposal_weights

    def reward(self, y, y_ast, score=None):
        """Return minus of edit distance.
        :param y:
        :param y_ast:
        :param score: edit distance
        :return:
        """
        if score is not None:
            return - score

            # TODO: calc and return edit distance of (y, y*)s
