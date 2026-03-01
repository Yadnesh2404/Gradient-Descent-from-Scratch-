# Gradient Descent from Scratch 📉

> **The goal here is not a high R² score.**
> It's to understand — line by line — how a model actually learns.

This project implements **Batch Gradient Descent** and **Stochastic Gradient Descent** from scratch using only NumPy. No optimizers, no `model.fit()` from sklearn, no shortcuts. Every gradient is derived by hand and applied manually.

The Diabetes dataset is used purely as a vessel — a real dataset to run the algorithm on. The metric results are secondary. What matters is seeing *why* the weights move the way they do.

---

## Project Structure

```
Gradient Descent/
├── batch_gadient_descent_from_scratch.py      # Batch GD — full dataset per update
├── stochastic_gadient_descent_from_scratch.py # SGD — one random sample per update
├── comparison.py                              # Animated live comparison of both
└── README.md
```

---

## The Core Idea

A linear regression model predicts:

```
ŷ = b₀ + b₁x₁ + b₂x₂ + ... + bₙxₙ
```

The model doesn't know the right values for `b₀...bₙ` upfront. It starts with a guess (zeros/ones) and then *iteratively corrects itself* using the gradient of the loss.

We use **Mean Squared Error** as the loss:

```
L = (1/n) × Σ (yᵢ - ŷᵢ)²
```

The gradient tells us: *in which direction is the loss increasing?* We step in the **opposite** direction to reduce it.

---

## Deriving the Gradients (the from-scratch part)

This is where the understanding lives. Rather than calling an optimizer, we work out the partial derivatives of the MSE loss ourselves.

**With respect to the intercept `b₀`:**
```
∂L/∂b₀ = -2 × mean(y - ŷ)
```

**With respect to each coefficient `bᵢ`:**
```
∂L/∂bᵢ = -2 × mean((y - ŷ) × xᵢ)
```

These plug directly into the weight update rule:
```
parameter = parameter - learning_rate × ∂L/∂parameter
```

No library computes this. It's written out explicitly in every update step.

---

## Two Variants — Same Math, Different Philosophy

### Batch Gradient Descent

Looks at the **entire dataset** before making a single weight update.

```python
for i in range(self.epochs):
    y_hat = self.intercept + np.dot(X_train, self.coef)  # all samples

    intercept_der = -2 * np.mean(y_train - y_hat)
    self.intercept = self.intercept - self.learning_rate * (intercept_der)

    coef_der = -2 * np.dot((y_train - y_hat), X_train) / X_train.shape[0]
    self.coef = self.coef - self.learning_rate * (coef_der)
```

Every gradient computed here is the "true" average gradient over all training samples. The loss decreases smoothly and predictably. It's the cleanest version to reason about.

---

### Stochastic Gradient Descent

Picks **one random sample** and updates immediately — then repeats.

```python
for i in range(self.epochs):
    for j in range(X_train.shape[0]):
        idx = np.random.randint(0, X_train.shape[0])
        y_hat = self.intercept + np.dot(X_train[idx], self.coef)

        intercept_der = -2 * (y_train[idx] - y_hat)
        self.intercept = self.intercept - self.learning_rate * (intercept_der)

        coef_der = -2 * np.dot((y_train[idx] - y_hat), X_train[idx])
        self.coef = self.coef - self.learning_rate * (coef_der)
```

Each gradient is computed from a single sample, which makes it noisy — but that noise is the entire point of studying it. The loss curve zig-zags, updates are erratic, yet the weights still converge. Understanding *why* that happens is more valuable than comparing final scores.

---

## Behavioral Differences (Not a Leaderboard)

| Property            | Batch GD                  | Stochastic GD               |
|---------------------|---------------------------|-----------------------------|
| Gradient computed from | Entire dataset         | 1 random sample             |
| Loss curve shape    | Smooth, monotone          | Noisy, erratic              |
| Update frequency    | Once per epoch            | Once per sample             |
| Learning rate       | `0.1`                     | `0.01` (lower due to noise) |
| Conceptual clarity  | High — easy to trace      | Lower — but more realistic  |

> The R² scores are printed at the end, but they aren't the measure of success here. A model tuned for 10,000 epochs with a better LR schedule would score higher — but that's not what this is about.

---

## What the Animation Shows

Running `comparison.py` opens a looping animated chart with 3 panels:

- **MSE Loss over Epochs** — the loss curve being drawn live, epoch by epoch. Batch GD drops steadily; SGD bounces around but trends down.
- **Step Size `|ΔLoss|` per Epoch** — how much the loss changes each epoch. Batch GD is disciplined; SGD is all over the place early on.
- **R² Score bars** — fill up in real time. Included to show the end result, not to judge it.

The animation loops continuously so you can observe the convergence pattern repeatedly.

```bash
python comparison.py
```

---

## Weight Initialization

```python
self.intercept = 0
self.coef = np.ones(X_train.shape[1])
```

Both models start at the same point — intercept at zero, all coefficients at one. This makes it easy to observe how the gradients steer the weights from this neutral starting position toward something meaningful.

---

## How to Run

```bash
pip install numpy scikit-learn matplotlib
```

```bash
# Animated comparison (main entry point)
python comparison.py

# Run Batch GD standalone
python batch_gadient_descent_from_scratch.py

# Run SGD standalone
python stochastic_gadient_descent_from_scratch.py
```

---

## Dataset

The **Diabetes dataset** (`sklearn.datasets.load_diabetes`):
- 442 samples, 10 normalized features
- Continuous target: disease progression score
- Split: 80% train / 20% test, `random_state=2`

Used here only because it's a clean, real-valued regression dataset. Any dataset would serve the same purpose — the dataset is not the point.

---

## Why from Scratch?

Using `sklearn`'s `LinearRegression` or any optimizer gives you a result. Building it yourself gives you an understanding. When you write out `∂L/∂b₀ = -2 × mean(y - ŷ)` and see the weights actually move because of it, the algorithm stops being a black box. That's the only goal of this project.
