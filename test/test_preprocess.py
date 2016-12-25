from preprocess import preprocess
import pytest


def test_features_scaling_invalid_arg_throw_exception():
    # GIVEN : input_data가 이차원이 아닌 경우
    datasets = [2, 3, 4, 6, 8, 9]

    # WHEN, THEN
    with pytest.raises(ValueError):
        preprocess.features_scaling(datasets)

    # GIVEN : input_data에 숫자가 아닌 다른 값이 있는 경우
    datasets = [[2, 3], [4, 6], [8, "A"]]

    # WHEN, THEN
    with pytest.raises(ValueError):
        preprocess.features_scaling(datasets)


def test_features_scaling_valid_arg_return_scaled_datasets():
    # GIVEN
    datasets = [[1, 3], [2, 4], [3, 5]]

    # WHEN
    scaled_datasets = preprocess.features_scaling(datasets)

    # THEN
    assert scaled_datasets == [[0.5, 0.75], [1.0, 1.0], [1.5, 1.25]]
