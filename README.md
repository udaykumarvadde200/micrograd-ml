# Micrograd ML

A from-scratch machine learning project built on top of a custom scalar automatic differentiation engine inspired by Andrej Karpathy's micrograd.

The project extends the basic autograd engine with additional mathematical operations, neural-network components, optimizers, gradient checking, and a real-world dataset experiment.

## Project Goals

- Understand automatic differentiation from first principles
- Implement forward and backward propagation manually
- Understand computational graphs
- Verify analytical gradients using numerical gradient checking
- Build a neural network without PyTorch or TensorFlow for the core training process
- Implement optimization algorithms from scratch
- Train the model on a real dataset
- Compare different optimizers experimentally

## Project Structure

```text
micrograd-ml/
├── micrograd/
│   ├── __init__.py
│   ├── engine.py
│   └── nn.py
├── datasets/
│   ├── __init__.py
│   └── breast_cancer/
│       ├── __init__.py
│       ├── load_data.py
│       └── preprocess.py
├── models/
│   └── __init__.py
├── optimizers/
│   ├── __init__.py
│   ├── sgd.py
│   └── adam.py
├── experiments/
│   └── breast_cancer/
│       ├── __init__.py
│       ├── train.py
│       └── loss_comparison.png
├── tests/
│   ├── __init__.py
│   ├── test_engine.py
│   └── test_optimizers.py
├── notebooks/
│   └── breast_cancer_experiment.ipynb
├── requirements.txt
├── setup.py
└── README.md
```

## 1. Automatic Differentiation Engine

The core of the project is the `Value` class in `micrograd/engine.py`.

Each `Value` object stores:

- numerical value
- gradient
- parent nodes
- operation that produced it
- backward function

For example:

```python
a = Value(2.0)
b = Value(3.0)
c = a * b
```

Calling:

```python
c.backward()
```

propagates gradients backward through the computational graph using the chain rule.

## 2. Supported Operations

The engine supports:

- Addition
- Subtraction
- Multiplication
- Division
- Power
- Negation
- ReLU
- Exponential
- Logarithm
- Sigmoid

These operations allow the engine to construct nonlinear models and losses.

## 3. Backpropagation

The `backward()` method performs reverse-mode automatic differentiation.

The process is:

```text
Forward computation
        ↓
Build computational graph
        ↓
Topological ordering
        ↓
Set output gradient = 1
        ↓
Traverse graph in reverse
        ↓
Apply local derivatives
        ↓
Accumulate gradients
```

Gradient accumulation is important when a variable contributes to the final output through multiple paths.

## 4. Neural Network

The neural-network components are implemented in `micrograd/nn.py`.

The project contains:

- `Module`
- `Neuron`
- `Layer`
- `MLP`

The experiment uses:

```python
model = MLP(30, [16, 8, 1], seed=42)
```

Architecture:

```text
30 input features
      ↓
16 neurons
      ↓
8 neurons
      ↓
1 output neuron
```

ReLU is used in the hidden layers. The final output is converted to a probability using sigmoid.

## 5. Loss Function

Binary Cross Entropy is used for binary classification:

```text
L = -[y log(p) + (1-y) log(1-p)]
```

where:

- `y` = target
- `p` = predicted probability

A small numerical-stability adjustment is used to prevent `log(0)` during training.

## 6. Dataset

The project uses the Breast Cancer Wisconsin dataset provided through scikit-learn.

The dataset contains:

```text
569 samples
30 features
2 classes
```

The data is split into training and test sets using:

```python
test_size=0.2
random_state=42
stratify=y
```

Feature standardization is performed using `StandardScaler`.

## 7. Gradient Checking

The project verifies analytical gradients from the custom autograd engine against numerical gradients.

Central difference is used:

```text
df/dx ≈ [f(x+h) - f(x-h)] / (2h)
```

Gradient checks cover:

- Exponential
- Sigmoid
- Logarithm
- ReLU
- Combined expressions
- Multiple variables
- Multiple computational paths
- MLP parameters

Current engine tests:

```text
11 passed
```

## 8. Optimizers

### SGD

The update rule is:

```text
θ ← θ - η∇θL
```

Implementation:

```text
optimizers/sgd.py
```

### Adam

Adam maintains first- and second-moment estimates and applies bias correction.

Implementation:

```text
optimizers/adam.py
```

The optimizer state is maintained separately for each parameter.

## 9. Training Experiment

The Breast Cancer dataset was used to compare SGD and Adam.

Both experiments used:

```text
Architecture: MLP(30, [16, 8, 1])
Initialization seed: 42
Training epochs: 50
Train/test split: random_state=42
```

Optimizer settings:

```text
SGD
learning rate = 0.01

Adam
learning rate = 0.001
```

The same model initialization and dataset split were used for both runs.

## 10. Results

### Test Accuracy

| Optimizer | Test Accuracy |
|-----------|---------------|
| SGD | 93.86% |
| Adam | 95.61% |

In this particular experiment, Adam achieved higher test accuracy than SGD.

The test set contains 114 samples:

```text
SGD  → 107 / 114 correct
Adam → 109 / 114 correct
```

### Training Loss

```text
SGD:
Epoch 0  → 0.3901
Epoch 10 → 0.0361
Epoch 20 → 0.0141
Epoch 30 → 0.0069
Epoch 40 → 0.0038

Adam:
Epoch 0  → 0.5185
Epoch 10 → 0.0404
Epoch 20 → 0.0145
Epoch 30 → 0.0023
Epoch 40 → 0.0003
```

The loss comparison plot is stored at:

```text
experiments/breast_cancer/loss_comparison.png
```

## 11. What This Project Demonstrates

```text
Mathematical operations
        ↓
Computational graph
        ↓
Automatic differentiation
        ↓
Backpropagation
        ↓
Neural network
        ↓
Binary cross entropy
        ↓
Gradient-based optimization
        ↓
Real dataset
        ↓
Experimental comparison
```

The core training process intentionally avoids high-level deep-learning frameworks such as PyTorch and TensorFlow.

## 12. Testing

Run engine tests:

```bash
python -m pytest tests/test_engine.py -v
```

Run optimizer tests:

```bash
python -m pytest tests/test_optimizers.py -v
```

## 13. Running the Experiment

From the project root:

```bash
python experiments/breast_cancer/train.py
```

The training script:

1. Loads the dataset
2. Splits and standardizes the data
3. Creates the MLP
4. Trains using SGD
5. Trains using Adam
6. Evaluates both models
7. Compares their results
8. Generates the training-loss plot

## 14. Key Learning Outcomes

Through this project, I implemented and investigated:

- Computational graphs
- Reverse-mode automatic differentiation
- Chain rule
- Gradient accumulation
- Numerical gradient checking
- Neural-network forward propagation
- Backpropagation
- ReLU and sigmoid
- Binary cross entropy
- Numerical stability
- SGD
- Adam
- Reproducible initialization
- Real-world dataset preprocessing
- Experimental comparison of optimizers

## Future Improvements

Potential extensions:

- Momentum optimizer
- Mini-batch training
- Learning-rate experiments
- Additional datasets
- More extensive optimizer comparisons
- Better experiment tracking
- Additional neural-network architectures
