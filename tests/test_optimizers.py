from micrograd.engine import Value
from optimizers.adam import Adam


def test_adam_single_parameter():
    x = Value(2.0)

    optimizer = Adam([x], lr=0.1)

    x.grad = 1.0

    old_value = x.data

    optimizer.step()

    assert x.data != old_value
    assert optimizer.t == 1
    assert optimizer.m[id(x)] != 0
    assert optimizer.v[id(x)] != 0

def test_adam_multiple_steps():
    x = Value(2.0)

    optimizer = Adam([x], lr=0.1)

    values = []

    for _ in range(5):
        x.grad = 1.0
        optimizer.step()
        values.append(x.data)

    assert optimizer.t == 5
    assert values[0] != values[-1]

from micrograd.engine import Value
from optimizers.momentum import Momentum


def test_momentum_single_parameter():
    x = Value(2.0)

    optimizer = Momentum(
        [x],
        lr=0.1,
        momentum=0.9
    )

    x.grad = 1.0

    old_value = x.data

    optimizer.step()

    assert x.data != old_value
    assert optimizer.velocity[id(x)] != 0


def test_momentum_multiple_steps():
    x = Value(2.0)

    optimizer = Momentum(
        [x],
        lr=0.1,
        momentum=0.9
    )

    values = []

    for _ in range(5):
        x.grad = 1.0
        optimizer.step()
        values.append(x.data)

    assert values[0] != values[-1]
    assert optimizer.velocity[id(x)] != 0