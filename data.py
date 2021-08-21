import numpy as np

class RelevanceJudgments:
    def __init__(self, labels_list, labels_range):
        '''
        :param label_list: [L, M, N] array, L: #queries, M: #documents per query, N: #assessors
        :param label_range: a int list, eg: 4-grade [0, 1, 2, 3]
        '''

        self.labels_list_raw = np.array(labels_list, dtype=np.int)
        self.labels_range = labels_range
        self.labels_range_dict = {}
        for i, label in enumerate(self.labels_range):
            self.labels_range_dict[label] = i

        self.L, self.M, self.N = self.labels_list_raw.shape
        self.G = len(labels_range) # grade scale

        self.labels_list = np.array([[[self.labels_range_dict[self.labels_list_raw[l, m, n]]
                              for n in range(self.N)]
                             for m in range(self.M)]
                            for l in range(self.L)])



class PreferenceJudgments:
    def __init__(self, judgments, relevancejudgments: RelevanceJudgments, SMOOTH_PARAM=1e-6):
        '''
        :param judgments: [H, 4] array,
        first col: the query id (range from 0 to L - 1)
        second col: the document id of left result (range from 0 to M - 1)
        third col: the document id of right result (range from 0 to M - 1)
        fourth col: preference result, < 0 / = 0 / > 0 represents left result is better / tied / worse than the right one
        :param SMOOTH_PARAM: param to exhibit zero-sample condition
        '''
        self.judgments = judgments
        self.relevancejudgments = relevancejudgments
        self.SMOOTH_PARAM = SMOOTH_PARAM

        self.gradelevel_preference_matrix_individual = None
        self.gradelevel_preference_matrix_aggregate = None
        self.documentlevel_preference_matrix_individual = None
        self.documentlevel_preference_matrix_aggregate = None


    def get_gradelevel_preference_matrix(self, mode):
        # mode: individual or aggregate
        if mode not in ['individual', 'aggregate']:
            assert 'invalid grade-level preference matrix mode'

        if mode == 'individual':
            if self.gradelevel_preference_matrix_individual is not None:
                return self.gradelevel_preference_matrix_individual
        elif mode == 'aggregate':
            if self.gradelevel_preference_matrix_aggregate is not None:
                return self.gradelevel_preference_matrix_aggregate

        grade_preference_counts_matrix = np.zeros([self.relevancejudgments.G, self.relevancejudgments.G], dtype=np.float)
        for judgment in self.judgments:
            qid, uidA, uidB, pscore = judgment
            scoresA, scoresB = self.relevancejudgments.labels_list[qid, uidA, :], \
                               self.relevancejudgments.labels_list[qid, uidB, :]
            if mode == 'individual':
                for i in range(len(scoresA)):
                    scoreA, scoreB = scoresA[i], scoresB[i]
                    if pscore < 0:
                        grade_preference_counts_matrix[scoreA, scoreB] += 1.
                    elif pscore == 0:
                        grade_preference_counts_matrix[scoreA, scoreB] += 0.5
                        grade_preference_counts_matrix[scoreB, scoreA] += 0.5
                    else:
                        grade_preference_counts_matrix[scoreB, scoreA] += 1.

            elif mode == 'aggregate':
                scoresA_sorted, scoresB_sorted = sorted(scoresA), sorted(scoresB)
                if len(scoresA_sorted) % 2 == 1:
                    idx_median = len(scoresA_sorted) // 2
                    scoreA, scoreB = scoresA_sorted[idx_median], scoresB_sorted[idx_median]
                    if pscore < 0:
                        grade_preference_counts_matrix[scoreA, scoreB] += 1.
                    elif pscore == 0:
                        grade_preference_counts_matrix[scoreA, scoreB] += 0.5
                        grade_preference_counts_matrix[scoreB, scoreA] += 0.5
                    else:
                        grade_preference_counts_matrix[scoreB, scoreA] += 1.
                else:
                    idx_medianR = len(scoresA_sorted) // 2
                    idx_medianL = idx_medianR - 1
                    scoreAL, scoreAR = scoresA_sorted[idx_medianL], scoresA_sorted[idx_medianR]
                    scoreBL, scoreBR = scoresB_sorted[idx_medianL], scoresB_sorted[idx_medianR]
                    if pscore < 0:
                        grade_preference_counts_matrix[scoreAL, scoreBL] += 0.25
                        grade_preference_counts_matrix[scoreAL, scoreBR] += 0.25
                        grade_preference_counts_matrix[scoreAR, scoreBL] += 0.25
                        grade_preference_counts_matrix[scoreAR, scoreBR] += 0.25
                    elif pscore == 0:
                        grade_preference_counts_matrix[scoreAL, scoreBL] += 0.125
                        grade_preference_counts_matrix[scoreAL, scoreBR] += 0.125
                        grade_preference_counts_matrix[scoreAR, scoreBL] += 0.125
                        grade_preference_counts_matrix[scoreAR, scoreBR] += 0.125
                        grade_preference_counts_matrix[scoreBL, scoreAL] += 0.125
                        grade_preference_counts_matrix[scoreBL, scoreAR] += 0.125
                        grade_preference_counts_matrix[scoreBR, scoreAL] += 0.125
                        grade_preference_counts_matrix[scoreBR, scoreAR] += 0.125
                    else:
                        grade_preference_counts_matrix[scoreBL, scoreAL] += 0.25
                        grade_preference_counts_matrix[scoreBL, scoreAR] += 0.25
                        grade_preference_counts_matrix[scoreBR, scoreAL] += 0.25
                        grade_preference_counts_matrix[scoreBR, scoreAR] += 0.25

        grade_preference_matrix = np.zeros([self.relevancejudgments.G, self.relevancejudgments.G],
                                                  dtype=np.float)
        for i in range(self.relevancejudgments.G):
            for j in range(self.relevancejudgments.G):
                grade_preference_matrix[i, j] = grade_preference_counts_matrix[i, j] / (grade_preference_counts_matrix[i, j]
                                                                                        + grade_preference_counts_matrix[j, i])

        if mode == 'individual':
            self.gradelevel_preference_matrix_individual = grade_preference_matrix
        elif mode == 'aggregate':
            self.gradelevel_preference_matrix_aggregate = grade_preference_matrix

        return grade_preference_matrix


    def get_documentlevel_preference_matrix(self, mode):
        # mode: individual or aggregate
        if mode not in ['individual', 'aggregate']:
            assert 'invalid grade-level preference matrix mode'

        if mode == 'individual':
            if self.documentlevel_preference_matrix_individual is not None:
                return self.documentlevel_preference_matrix_individual
        elif mode == 'aggregate':
            if self.documentlevel_preference_matrix_aggregate is not None:
                return self.documentlevel_preference_matrix_aggregate

        if mode == 'individual' and self.gradelevel_preference_matrix_individual is None:
            self.gradelevel_preference_matrix_individual = self.get_gradelevel_preference_matrix('individual')
        elif mode == 'aggregate' and self.gradelevel_preference_matrix_aggregate is None:
            self.gradelevel_preference_matrix_aggregate = self.get_gradelevel_preference_matrix('aggregate')

        document_preference_matrix = np.zeros([self.relevancejudgments.L, self.relevancejudgments.M, self.relevancejudgments.M], dtype=np.float)

        for i in range(self.relevancejudgments.L):
            for j in range(self.relevancejudgments.M):
                document_preference_matrix[i, j, j] = 0.5

                scoresJ = self.relevancejudgments.labels_list[i, j, :]
                for k in range(j):
                    scoresK = self.relevancejudgments.labels_list[i, k, :]
                    if mode == 'individual':
                        _prob = 0.
                        for o in range(len(scoresJ)):
                            _prob += self.gradelevel_preference_matrix_individual[scoresJ[o], scoresK[o]]
                        _prob /= float(len(scoresJ))
                        document_preference_matrix[i, j, k] = _prob
                        document_preference_matrix[i, k, j] = 1. - _prob
                    elif mode == 'aggregate':
                        scoresJ_sorted, scoresK_sorted = sorted(scoresJ), sorted(scoresK)
                        if len(scoresJ_sorted) % 2 == 1:
                            idx_median = len(scoresJ_sorted) // 2
                            document_preference_matrix[i, j, k] = self.gradelevel_preference_matrix_aggregate[scoresJ_sorted[idx_median], scoresK_sorted[idx_median]]
                            document_preference_matrix[i, k, j] = self.gradelevel_preference_matrix_aggregate[
                                scoresK_sorted[idx_median], scoresJ_sorted[idx_median]]
                        else:
                            idx_medianR = len(scoresJ_sorted) // 2
                            idx_medianL = idx_medianR - 1
                            document_preference_matrix[i, j, k] = (self.gradelevel_preference_matrix_aggregate[scoresJ_sorted[idx_medianL], scoresK_sorted[idx_medianL]] +
                                                                   self.gradelevel_preference_matrix_aggregate[scoresJ_sorted[idx_medianL], scoresK_sorted[idx_medianR]] +
                                                                   self.gradelevel_preference_matrix_aggregate[scoresJ_sorted[idx_medianR], scoresK_sorted[idx_medianL]]+
                                                                   self.gradelevel_preference_matrix_aggregate[scoresJ_sorted[idx_medianR], scoresK_sorted[idx_medianR]]) / 4.
                            document_preference_matrix[i, k, j] = 1. - document_preference_matrix[i, j, k]

        if mode == 'individual':
            self.documentlevel_preference_matrix_individual = document_preference_matrix
        elif mode == 'aggregate':
            self.documentlevel_preference_matrix_aggregate = document_preference_matrix

        return document_preference_matrix

    def reset(self):
        self.gradelevel_preference_matrix_individual = None
        self.gradelevel_preference_matrix_aggregate = None
        self.documentlevel_preference_matrix_individual = None
        self.documentlevel_preference_matrix_aggregate = None



if __name__ == '__main__':
    data_4grade = np.array([[[3, 3, 3], [2, 3, 3], [2, 2, 2], [3, 3, 3], [3, 3, 3]]])
    rj = RelevanceJudgments(data_4grade, [3, 2, 1, 0])
    print(rj.labels_list)
