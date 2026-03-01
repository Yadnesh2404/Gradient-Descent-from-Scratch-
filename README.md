# Gradient Descent from Scratch 📉

An implementation of **Batch Gradient Descent** and **Stochastic Gradient Descent** built entirely from scratch using only NumPy — no `sklearn` optimizers, no black boxes. The goal is to deeply understand how a linear regression model actually *learns* by manually deriving and applying gradients at every step.

Tested on the **Diabetes dataset** from `sklearn.datasets`.

---

## Project Structure

```
Gradient Descent/
├── batch_gadient_descent_from_scratch.py      # Batch GD implementation
├── stochastic_gadient_descent_from_scratch.py # Stochastic GD implementation
├── comparison.py                              # Animated live comparison of both
└── README.md
```

---

## The Algorithm — Built from Scratch

### What are we optimizing?

We're fitting a **Linear Regression** model of the form:

```
ŷ = b₀ + b₁x₁ + b₂x₂ + ... + bₙxₙ
```

Where `b₀` is the intercept and `b₁...bₙ` are the feature coefficients. The model learns these values by minimizing the **Mean Squared Error (MSE) loss**:

```
L = (1/n) * Σ (yᵢ - ŷᵢ)²
```

The lower the loss, the better the model's predictions.

---

### How Gradient Descent Works

Instead of solving for the optimal weights analytically (which is expensive for large datasets), Gradient Descent iteratively nudges the weights in the direction that reduces the loss.

At each step, we compute the **partial derivative of the loss** with respect to each parameter and update them:

```
parameter = parameter - learning_rate × ∂L/∂parameter
```

**Deriving the gradients from MSE:**

For the intercept `b₀`:
```
∂L/∂b₀ = -2 × mean(y - ŷ)
```

For the coefficients `bᵢ`:
```
∂L/∂bᵢ = -2 × mean((y - ŷ) × xᵢ)
```

These are implemented directly in code — no autodiff, no library magic.

---

## Batch GD vs Stochastic GD

Both variants use the same math, but differ in *how much data* they look at per update.

### Batch Gradient Descent

Uses the **entire training dataset** to compute gradients at each step.

```python
for i in range(self.epochs):
    y_hat = self.intercept + np.dot(X_train, self.coef)   # predict on ALL samples

    intercept_der = -2 * np.mean(y_train - y_hat)
    self.intercept = self.intercept - self.learning_rate * (intercept_der)

    coef_der = -2 * np.dot((y_train - y_hat), X_train) / X_train.shape[0]
    self.coef = self.coef - self.learning_rate * (coef_der)
```

- ✅ Smooth, stable loss curve
- ✅ Guaranteed to head towards the minimum every step
- ❌ Slow on large datasets — every update requires a full pass

---

### Stochastic Gradient Descent

Uses a **single random sample** per update, looping through the whole dataset each epoch.

```python
for i in range(self.epochs):
    for j in range(X_train.shape[0]):
        idx = np.random.randint(0, X_train.shape[0])      # pick ONE random sample
        y_hat = self.intercept + np.dot(X_train[idx], self.coef)

        intercept_der = -2 * (y_train[idx] - y_hat)
        self.intercept = self.intercept - self.learning_rate * (intercept_der)

        coef_der = -2 * np.dot((y_train[idx] - y_hat), X_train[idx])
        self.coef = self.coef - self.learning_rate * (coef_der)
```

- ✅ Much faster updates — learns from one sample at a time
- ✅ The noise can help escape shallow local minima
- ❌ Noisy loss curve — doesn't decrease monotonically

---

### Side-by-Side Comparison

| Property              | Batch GD          | Stochastic GD       |
|-----------------------|-------------------|---------------------|
| Data per update       | Full dataset      | 1 random sample     |
| Loss curve            | Smooth            | Noisy / jagged      |
| Convergence stability | High              | Low (but often better final result) |
| Learning rate used    | `0.1`             | `0.01` (lower, due to noisy updates) |
| Final R² (100 epochs) | ~0.10             | ~0.44               |

> SGD's noise actually helps it explore the loss surface better, often landing at a lower final loss than Batch GD with the same number of epochs.

---

## Weight Initialization

Both models initialize weights the same way:

```python
self.intercept = 0
self.coef = np.ones(X_train.shape[1])   # all coefficients start at 1
```

Starting at zero/ones gives a neutral baseline. The gradients then steer the weights toward the minimum of the loss surface.

---

## Live Animated Comparison

Running `comparison.py` opens a **looping animated plot** with 3 panels:

- **MSE Loss over Epochs** — watch both models' loss drop in real time
- **Step Size `|ΔLoss|` per Epoch** — shows how large each update is; BGD is steady, SGD is erratic early on
- **R² Score bars** — fill up to their final values as training progresses

```bash
python comparison.py
```

The animation loops continuously (like a GIF) with a 1.5s pause between runs.

---

## How to Run

**Requirements:**
```bash
pip install numpy scikit-learn matplotlib
```

**Batch GD alone:**
```bash
python batch_gadient_descent_from_scratch.py
```

**Stochastic GD alone:**
```bash
python stochastic_gadient_descent_from_scratch.py
```

**Animated comparison (main file):**
```bash
python comparison.py
```

---

## Dataset

The **Diabetes dataset** (`sklearn.datasets.load_diabetes`) contains:
- **442 samples**, **10 normalized features** (age, BMI, blood pressure, etc.)
- Continuous target: disease progression one year after baseline
- Split: **80% train / 20% test** with `random_state=2`

---

## Key Takeaway

Gradient Descent is at the heart of almost every machine learning algorithm. By implementing it from scratch — manually writing out the partial derivatives of the loss, updating each parameter by hand, and tracking convergence — the mechanics become fully transparent. There's no optimizer doing the work behind the scenes; every weight update is explicitly computed and applied.
