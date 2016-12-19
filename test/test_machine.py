from ml import machine
import types
import pytest


def test_gradient_descent_invalid_arg_throw_error():
    # GIVEN : input_data가 이차원이 아닌 경우
    input_dataset = [2, 3, 4, 6, 8, 9]
    output_dataset = [7, 14, 25]

    # WHEN, THEN
    with pytest.raises(ValueError):
        machine.gradient_descent(input_dataset, output_dataset, 0.01)


def test_gradient_descent_valid_arg_return_function():
    # GIVEN
    input_dataset = [[2, 4], [3, 6]]
    output_dataset = [7, 14]

    # WHEN
    linear_f = machine.gradient_descent(input_dataset, output_dataset, 0.01, step=14000)

    # THEN
    assert isinstance(linear_f, types.FunctionType)
    assert machine.rmse(input_dataset, output_dataset, linear_f) <= 0.01
