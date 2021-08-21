# the toy model of Figure 1 in Paper "Evaluating Relevance Judgments with Pairwise Discriminative Power"

import numpy as np

np.random.seed(2021)

import krippendorff
from util import PDP
from data import RelevanceJudgments
from data import PreferenceJudgments

if __name__ == '__main__':
    P4g = np.array([[0.5, 0.15, 0.10, 0.05], [0.85, 0.5, 0.15, 0.1], [0.9, 0.85, 0.5, 0.15], [0.95, 0.9, 0.85, 0.5]])
    P7g = np.array([[0.5, 0.2, 0.15, 0.1, 0.06, 0.03, 0.01],
                    [0.8, 0.5, 0.2, 0.15, 0.1, 0.06, 0.03],
                    [0.85, 0.8, 0.5, 0.2, 0.15, 0.1, 0.06],
                    [0.9, 0.85, 0.8, 0.5, 0.2, 0.15, 0.1],
                    [0.94, 0.9, 0.85, 0.8, 0.5, 0.2, 0.15],
                    [0.97, 0.94, 0.9, 0.85, 0.8, 0.5, 0.2],
                    [0.99, 0.97, 0.94, 0.9, 0.85, 0.8, 0.5]])

    data_4grade = np.array(
        [[[3, 3, 3], [2, 3, 3], [2, 2, 2], [3, 3, 3], [3, 3, 3]]])  # indi: 4.182342863136126, agg: 4.187759589145479
    # indi: 4.2454, agg: 4.1978
    data_7grade = np.array([[[5, 6, 5], [4, 5, 5], [3, 5, 3], [6, 6, 6], [5, 5, 4]]])  # indi: 4.0443, agg: 3.9009

    alpha_4grade = krippendorff.alpha(np.array(data_4grade[0]).transpose(), level_of_measurement='ordinal')
    alpha_7grade = krippendorff.alpha(np.array(data_7grade[0]).transpose(), level_of_measurement='ordinal')
    print(f'Krippendorff\'s alpha of 4-grade = {alpha_4grade}, alpha of 7-grade = {alpha_7grade}')
    # 4-grade: 0.6818, 7-grade: 0.5713

    relevancejudgment_grade4 = RelevanceJudgments(data_4grade, [0, 1, 2, 3])
    preferencejudgment_grade4 = PreferenceJudgments(None, relevancejudgment_grade4)
    preferencejudgment_grade4.gradelevel_preference_matrix_individual = P4g # set the grade level preference matrix automatically
    preferencejudgment_grade4.gradelevel_preference_matrix_aggregate = P4g

    relevancejudgment_grade7 = RelevanceJudgments(data_7grade, [0, 1, 2, 3, 4, 5, 6])
    preferencejudgment_grade7 = PreferenceJudgments(None, relevancejudgment_grade7)
    preferencejudgment_grade7.gradelevel_preference_matrix_individual = P7g
    preferencejudgment_grade7.gradelevel_preference_matrix_aggregate = P7g


    pdp = PDP()

    print(f'PDP of 7-grade data in individual = {pdp(preferencejudgment_grade7, "individual")[0]}') # 4.0249
    print(f'PDP of 7-grade data in aggregate = {pdp(preferencejudgment_grade7, "aggregate")[0]}') # 3.8180
    print(f'PDP of 4-grade data in individual = {pdp(preferencejudgment_grade4, "individual")[0]}') # 4.2142
    print(f'PDP of 4-grade data in aggregate = {pdp(preferencejudgment_grade4, "aggregate")[0]}') # 4.1705
