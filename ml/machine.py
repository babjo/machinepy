import numbers
import numpy as np
import math


def gradient_descent(input_dataset, output_dataset, learning_rate, step=100):
    validate_args(input_dataset, output_dataset, learning_rate)

    # [[x, x]
    #  [x, x]
    #  [x, x]]
    input_matrix = np.array(input_dataset)
    m = input_matrix.shape[0]

    # [[1]
    #  [1]
    #  [1]]
    x_zeros = np.array([[1] * m]).T

    # [[1, x, x]
    #  [1, x, x]
    #  [1, x, x]]
    input_matrix = np.hstack((x_zeros, input_matrix))

    # [[output_0],
    #  [output_1],
    #  [output_2],
    #  [output_3]]
    output_vector = np.array([output_dataset]).T

    # [[theta_0],
    #  [theta_1],
    #  [theta_2]]
    theta_vector = np.random.rand(input_matrix.shape[1], 1)

    for _ in range(step):
        updated_theta = []
        for j in range(len(theta_vector)):
            updated_theta_j = theta_vector[j] - learning_rate * 1 / m * np.dot(input_matrix[:, j].T, (
                np.dot(input_matrix, theta_vector) - output_vector))
            updated_theta.append(updated_theta_j)
        theta_vector = np.array(updated_theta)

    return lambda input_x: np.dot(np.array([1] + input_x), theta_vector)


def rmse(input_dataset, output_dataset, linear_f):
    return math.sqrt(sum(
        [(output_dataset[index] - linear_f(input_data)) ** 2 for index, input_data in enumerate(input_dataset)]) / len(
        input_dataset))


def validate_args(input_dataset, output_dataset, learning_rate=0.1):
    if not isinstance(input_dataset, list):
        raise ValueError
    if not isinstance(output_dataset, list):
        raise ValueError
    for input_data in input_dataset:
        if not isinstance(input_data, list):
            raise ValueError
        for value in input_data:
            if not isinstance(value, numbers.Number):
                raise ValueError
    for value in output_dataset:
        if not isinstance(value, numbers.Number):
            raise ValueError
    if not isinstance(learning_rate, numbers.Number):
        raise ValueError


def normal_equation(input_dataset, output_dataset):
    validate_args(input_dataset, output_dataset)
    # θ=(XT X)−1 XT y

    # [[x, x]
    #  [x, x]
    #  [x, x]]
    input_matrix = np.array(input_dataset)
    m = input_matrix.shape[0]

    # [[1]
    #  [1]
    #  [1]]
    x_zeros = np.array([[1] * m]).T

    # [[1, x, x]
    #  [1, x, x]
    #  [1, x, x]]
    input_matrix = np.hstack((x_zeros, input_matrix))

    # [[output_0],
    #  [output_1],
    #  [output_2],
    #  [output_3]]
    output_vector = np.array([output_dataset]).T

    # [[theta_0],
    #  [theta_1],
    #  [theta_2]]

    a = np.dot(input_matrix.T, input_matrix)
    if is_singular(a):
        raise ValueError

    b = np.dot(input_matrix.T, output_vector)
    theta_vector = np.linalg.solve(a, b)

    return lambda input_x: np.dot(np.array([1] + input_x), theta_vector)


def is_singular(target):
    return np.linalg.det(target) == 0
