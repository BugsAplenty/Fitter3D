#!/usr/bin/env python3

import numpy as np


def softposit(image_points, model_points):
    """
    Takes pointcloud extracted from a 3D model of an object and pointmap extracted from a 2D image of the same object
    and calculates the optimal transpose between them. :param image_points: :param model_points: :return:
    """
    # Initialize.
    alpha = 0.0001
    beta_init = 0.004
    beta_final = 0.1
    beta_n_iters = 10
    model_points_homogenous = np.concatenate((self.model_points, np.ones((np.size(self.model_points, 0), 1))),
                                             axis=1)
    transpose_homogenous = np.array([np.concatenate((rmat[0].reshape(-1, 1),
                                                     tvec[2].reshape(-1, 1))).squeeze(),
                                     np.concatenate((rmat[1].reshape(-1, 1),
                                                     tvec[1].reshape(-1, 1))).squeeze()])
    # TODO: Rotations of the axes are mixed up. Need to test later.
    w = np.ones(np.size(image_points, 0))
    for beta in np.linspace(beta_init, beta_final, beta_n_iters):
        points_dist = np.dot(model_points_homogenous, transpose_homogenous)
        correspondence = np.exp(-beta * (points_dist - alpha))
        correspondence_normalized = sinkhorn(correspondence)
        correspondence_normalized_row_sums = np.sum(correspondence_normalized, axis=1).reshape(-1, 1)[0:-1, :]


def slack(matrix):
    matrix_slack_cols = np.concatenate((matrix, np.ones((np.size(matrix, 0), 1))), axis=1)
    matrix_slack = np.concatenate((matrix_slack_cols, np.ones((1, np.size(matrix_slack_cols,
                                                                          1)))), axis=0)
    return matrix_slack


def sinkhorn(correspondence):
    max_iter_sinkhorn = 60
    termination_threshold = 0.001
    correspondence_difference = 1 + termination_threshold
    i = 0
    correspondence_slack = slack(correspondence)
    while (correspondence_difference > termination_threshold) and (i < max_iter_sinkhorn):
        correspondence_slack_previous = correspondence_slack
        correspondence_col_sums = np.sum(correspondence_slack, axis=0).reshape(-1, 1).transpose()
        correspondence_col_sums[:, -1] = 1
        correspondence_col_sums = np.repeat(correspondence_col_sums, np.size(correspondence_slack, 0), axis=0)
        correspondence_slack = np.divide(correspondence_slack, correspondence_col_sums)

        correspondence_row_sums = np.sum(correspondence_slack, axis=1).reshape(-1, 1)
        correspondence_row_sums[-1] = 1
        correspondence_row_sums = np.repeat(correspondence_row_sums, np.size(correspondence_slack, 1), axis=1)
        correspondence_slack = np.divide(correspondence_slack, correspondence_row_sums)

        correspondence_difference = np.sum(np.abs(correspondence_slack - correspondence_slack_previous))
        i += 1

    return correspondence_slack
