import torch
from micrograd.engine import Value

def numerical_gradient(x, f, h=1e-4):
    original_value = x.data

    x.data = original_value + h
    y_plus = f().data

    x.data = original_value - h
    y_minus = f().data

    x.data = original_value

    return (y_plus - y_minus) / (2 * h)
    
def test_sanity_check():

    x = Value(-4.0)
    z = 2 * x + 2 + x
    q = z.relu() + z * x
    h = (z * z).relu()
    y = h + q + q * x
    y.backward()
    xmg, ymg = x, y

    x = torch.Tensor([-4.0]).double()
    x.requires_grad = True
    z = 2 * x + 2 + x
    q = z.relu() + z * x
    h = (z * z).relu()
    y = h + q + q * x
    y.backward()
    xpt, ypt = x, y

    # forward pass went well
    assert ymg.data == ypt.data.item()
    # backward pass went well
    assert xmg.grad == xpt.grad.item()

def test_more_ops():

    a = Value(-4.0)
    b = Value(2.0)
    c = a + b
    d = a * b + b**3
    c += c + 1
    c += 1 + c + (-a)
    d += d * 2 + (b + a).relu()
    d += 3 * d + (b - a).relu()
    e = c - d
    f = e**2
    g = f / 2.0
    g += 10.0 / f
    g.backward()
    amg, bmg, gmg = a, b, g

    a = torch.Tensor([-4.0]).double()
    b = torch.Tensor([2.0]).double()
    a.requires_grad = True
    b.requires_grad = True
    c = a + b
    d = a * b + b**3
    c = c + c + 1
    c = c + 1 + c + (-a)
    d = d + d * 2 + (b + a).relu()
    d = d + 3 * d + (b - a).relu()
    e = c - d
    f = e**2
    g = f / 2.0
    g = g + 10.0 / f
    g.backward()
    apt, bpt, gpt = a, b, g

    tol = 1e-6
    # forward pass went well
    assert abs(gmg.data - gpt.data.item()) < tol
    # backward pass went well
    assert abs(amg.grad - apt.grad.item()) < tol
    assert abs(bmg.grad - bpt.grad.item()) < tol


def test_gradient_check_exp():
    x = Value(2.0)

    y = x.exp()
    y.backward()

    numerical = numerical_gradient(
        x,
        lambda: x.exp()
    )

    assert abs(x.grad - numerical) < 1e-5

def test_gradient_check_sigmoid():
    x = Value(0.7)

    y = x.sigmoid()
    y.backward()

    numerical = numerical_gradient(
        x,
        lambda: x.sigmoid()
    )

    assert abs(x.grad - numerical) < 1e-5


def test_gradient_check_log():
    x = Value(2.0)

    y = x.log()
    y.backward()

    numerical = numerical_gradient(
        x,
        lambda: x.log()
    )

    assert abs(x.grad - numerical) < 1e-5


def test_gradient_check_relu():
    x = Value(2.0)

    y = x.relu()
    y.backward()

    numerical = numerical_gradient(
        x,
        lambda: x.relu()
    )

    assert abs(x.grad - numerical) < 1e-5


def test_gradient_check_combined():
    x = Value(0.7)

    y = (x * x + x).sigmoid()
    y.backward()

    numerical = numerical_gradient(
        x,
        lambda: (x * x + x).sigmoid()
    )

    assert abs(x.grad - numerical) < 1e-5
    
def test_gradient_check_multiple_variables():
    a = Value(2.0)
    b = Value(3.0)

    y = (a * b + a).sigmoid()
    y.backward()

    numerical_a = numerical_gradient(
        a,
        lambda: (a * b + a).sigmoid()
    )

    numerical_b = numerical_gradient(
        b,
        lambda: (a * b + a).sigmoid()
    )

    assert abs(a.grad - numerical_a) < 1e-5
    assert abs(b.grad - numerical_b) < 1e-5

def test_gradient_accumulation():
    x = Value(3.0)

    y = x * x + x
    y.backward()

    numerical = numerical_gradient(
        x,
        lambda: x * x + x
    )

    assert abs(x.grad - numerical) < 1e-5

def test_gradient_accumulation_multiple_paths():
    x = Value(3.0)

    y = x * x + x * x + x
    y.backward()

    numerical = numerical_gradient(
        x,
        lambda: x * x + x * x + x
    )

    assert abs(x.grad - numerical) < 1e-5

def test_gradient_check_mlp():
    from micrograd.nn import MLP

    mlp = MLP(2, [3, 1])

    x = [Value(0.5), Value(-1.0)]

    y = mlp(x)

    y.backward()

    for p in mlp.parameters():
        numerical = numerical_gradient(
            p,
            lambda: mlp(x)
        )

        assert abs(p.grad - numerical) < 1e-5