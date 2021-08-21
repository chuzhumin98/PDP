import numpy as np
import math

from data import RelevanceJudgments
from data import PreferenceJudgments

class PDP:
    def __init__(self, THRESHOLD_PREFERENCE=0.01, ITER_MAX_SCORE_FUNCTION=1000, BATCHSIZE_SCORE_FUNCTION=20,
                 LR_SCORE_FUNCTION=0.1, LAMDA_SCORE_FUNCTION=0, SAMPLE_SIZE=50000, SCORE_FUNCTION_W0=None):
        self.THRESHOLD_PREFERENCE = THRESHOLD_PREFERENCE
        self.ITER_MAX_SCORE_FUNCTION = ITER_MAX_SCORE_FUNCTION
        self.BATCHSIZE_SCORE_FUNCTION = BATCHSIZE_SCORE_FUNCTION
        self.LR_SCORE_FUNCTION = LR_SCORE_FUNCTION
        self.LAMDA_SCORE_FUNCTION = LAMDA_SCORE_FUNCTION
        self.SAMPLE_SIZE = SAMPLE_SIZE
        self.SCORE_FUNCTION_W0 = SCORE_FUNCTION_W0

    def __call__(self, preferencejudgments: PreferenceJudgments, mode, *args, **kwargs):
        return self.calculate_PDP(preferencejudgments, mode)

    def calculate_PDP(self, preferencejudgments: PreferenceJudgments, mode):
        if mode not in ['individual', 'aggregate']:
            assert 'invalid grade-level preference matrix mode'

        document_preference_matrix = preferencejudgments.get_documentlevel_preference_matrix(mode)
        L = document_preference_matrix.shape[0]
        score_vec = np.array([self.score_function(document_preference_matrix[i], W0=self.SCORE_FUNCTION_W0) for i in range(L)])
        return [self._calculate_PDP(score_vec[i]) for i in range(L)]

    def calculate_PDP_with_documentpreferencematrix(self, document_preference_matrix):
        score_vec = self.score_function(document_preference_matrix, W0=self.SCORE_FUNCTION_W0)
        return self._calculate_PDP(score_vec)


    def _calculate_PDP(self, scores_vec):
        probs = np.exp(scores_vec)
        N = len(scores_vec)
        logPs = []
        for _ in range(self.SAMPLE_SIZE):
            probs_norm = probs / np.sum(probs)
            rank_list = np.random.choice(N, N, p=probs_norm, replace=False)
            probs_copy = np.copy(probs)

            logP = 0.

            for j in range(N):
                index = rank_list[j]
                p_j = probs_copy[index] / np.sum(probs_copy)
                probs_copy[index] = 0.
                logP -= math.log(p_j)

            logPs.append(logP)

        return np.mean(logPs)


    def score_function(self, document_preference_matrix, W0=None):
        def set_threshod(preference_matrix):
            preference_matrix_adjusted = np.minimum(preference_matrix, 1 - self.THRESHOLD_PREFERENCE)
            preference_matrix_adjusted = np.maximum(preference_matrix_adjusted, self.THRESHOLD_PREFERENCE)
            return preference_matrix_adjusted

        document_preference_matrix = set_threshod(document_preference_matrix)
        xs, ys = self.documentpreferencematrix2samples(document_preference_matrix)
        scores_vec = self._train_score_function(xs, ys, W0)
        return scores_vec


    def _train_score_function(self, xs, ys, W0=None):
        N, n = xs.shape[0], xs.shape[1]
        if W0 is not None and N != len(W0):
            assert 'W0 is not xs.shape[0] length array'

        if W0 is not None:
            W = np.copy(W0)
        else:
            W = np.random.randn(n)


        for _ in range(self.ITER_MAX_SCORE_FUNCTION):
            indexes = np.array(range(N))
            np.random.shuffle(indexes)
            batchsize = self.BATCHSIZE_SCORE_FUNCTION
            split_num = N // batchsize
            for j in range(split_num):
                low, high = j * batchsize, (j + 1) * batchsize
                xs_batch, ys_batch = xs[indexes[low:high]], ys[indexes[low:high]]
                _, gradient, _ = self.softmax_classifier(W, xs_batch, ys_batch, lamda=self.LAMDA_SCORE_FUNCTION)
                W = W - self.LR_SCORE_FUNCTION * gradient

        return W



    def documentpreferencematrix2samples(self, preference_matrix):
        """
        Inputs:
        - preference_matrix: [N, N]

        Returns:
        - xs: numpy, [N*(N-1), N]
        - ys: numpy, [N*(N-1), N]
        """
        N = preference_matrix.shape[0]
        xs = np.zeros([N * (N - 1), N], dtype=np.float64)
        ys = np.zeros([N * (N - 1)], dtype=np.float64)
        cnt = 0
        for i in range(N):
            for j in range(N):
                if i != j:
                    xs[cnt, i] = 1
                    xs[cnt, j] = -1
                    ys[cnt] = preference_matrix[i, j]
                    cnt += 1
        return xs, ys

    def softmax_classifier(self, W, input, label, lamda=0.5):
        N = len(input)
        D = len(W)

        # get the probs matrix
        probs = np.exp(np.dot(input, W)).reshape(
            -1)  # shape (N), h_k^{(n)}, the probability of sample n belongs to class k
        probs = probs / (1 + probs)

        # calculate the loss
        errors = label * np.log(probs) + (1 - label) * np.log(1 - probs)
        loss = np.mean(np.sum(errors)) + lamda * np.sum(np.multiply(W, W)) / 2

        # calculate gradient
        gradient = np.mean(np.transpose(np.tile(probs - label, [D, 1])) * input, axis=0) + lamda * W

        # calculate prediction
        prediction = probs

        return loss, gradient, prediction


