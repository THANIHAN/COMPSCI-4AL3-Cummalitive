CS/SE 4AL3 · Applications of Machine Learning · Lectures 1–12

# 4AL3 Midterm Study Guide

Everything from the first five weeks of slides, rebuilt as notes you can learn from: the ideas, every formula, every calculation done in class (re-checked), and new practice problems in the same style.

Midterm: Oct 22 In class · MDCL 1305 45 minutes 4 computation questions 1 handwritten cheat sheet McMaster standard calculator

**Contents**0 / 14 studied

[✓What ML isL1–2](#big-picture) [✓kNN & hyperparametersL3](#knn) [✓Linear regression & OLSL3–4](#linreg) [✓Gradient descentL4–5](#gd) [✓GeneralizationL4–6](#generalize) [✓Logistic regressionL5–6](#logreg) [✓SGD & convexityL6–7](#sgd) [✓Softmax (multinomial)L7–8](#softmax) [✓Evaluation metricsL7–8](#eval) [✓Features & regularizationL8–9](#reg) [✓Support vector machinesL10](#svm) [✓SVM training & kernelsL11](#kernels) [✓Data preprocessingL11](#prep) [✓Decision trees & ensemblesL12](#trees)

[Practice problems13](#practice) [Concept quiz20](#quiz) [Formula sheet](#sheet)

## How to use this

- Work through the 14 units in order; each one ends with a few *check yourself* questions. Press **Mark studied** when you're done and the contents rail keeps your place.
- Boxes labelled IN-CLASS CALCULATION redo the exact numbers from the slides. Prof. Welch said these matter most: *"any calculation we did in class."* Rework each one on paper before reading the steps.
- Then do the [practice problems](#practice) (fresh numbers, answer boxes check you) and the [concept quiz](#quiz).
- The [formula sheet](#sheet) is a checklist for your cheat sheet. The course rule is that the sheet must be in your own handwriting, so copy what you need by hand.

Lectures 1–2

## What machine learning is

An ML model is a function that maps features to labels, with internal parameters adjusted so it does well on future, unseen data.

### Definitions from the slides

- **Statistical learning**: a set of tools for modeling and understanding complex datasets; a subfield of statistics focused on supervised and unsupervised modeling and prediction.
- **Machine learning algorithm**: a system that adapts some internal parameters of a predictor so it performs well on future unseen input data.
- **Deep learning**: learning a hierarchy of concepts, each defined through simpler ones. Drawn as a graph, it is deep (many layers).
- The field sits where *math* (stats, calculus, linear algebra) meets *software engineering* (algorithms, dev, Python), feeding both data science and AI applications.
- **AI is not just chatbots.** The Turing test (Turing, 1950) is passed if an interrogator can't tell the machine from the human.

### Three types of learning

| Type | Data | Feedback | Goal |
| --- | --- | --- | --- |
| **Supervised** | Labeled examples | Direct (correct label) | Predict an outcome. Classification (discrete) or regression (continuous). |
| **Unsupervised** | No labels | None | Find hidden structure, e.g. clustering using distances between vector representations. |
| **Reinforcement** | Environment with states | Reward function | Learn a *policy*: which action to take in a given state, over a sequence of actions. Can learn funny, unintended behaviour. |

### The supervised loop (fruit example)

Show the model examples ("Is it fruit?" yes/no). When it's wrong, provide the correct label and readjust. Then classify new instances "in the wild."

Two traps the slides call out. "I solved fruit classification" really means: *we have a function mapping features to fruit/not-fruit, based on one definition of fruit and one sample of instances*. And "the model understands fruit" attributes human qualities to a function that outputs a label.

### Operationalization

Before you can learn something you must define how it's measured. **Fruit** → "is it in the list of fruits?"; **cognitive load** → ability to remember a sequence of numbers; **subject knowledge** → test scores; **empathy** → assessed cognitive/emotional behaviour in language. Model quality is capped by data quality: *garbage in, garbage out*.

### Dataset vocabulary (iris example)

| Part of the table | Synonyms you'll see |
| --- | --- |
| Row | sample = example = data point = item = instance |
| Column | feature = attribute = property = dimension |
| What you predict | label = class = category = target |

Iris: inputs sepal length, sepal width, petal length, petal width → output species. The features you choose can have a huge impact.

Grouping customers by purchase history when nobody has labelled them: which type of learning?

Unsupervised (clustering). No labels, no feedback; we organize instances by distance between their feature vectors.

A game agent gets +1 for each coin and learns which move to make in each situation. What is "the model"?

A **policy**: a mapping from state to action, learned from a reward function (reinforcement learning).

Why is "operationalization" a modeling decision and not a detail?

The label definition is what the model actually learns. If "empathy" is measured by a particular annotation of language, the model learns that annotation, with all its limits.

Lecture 3

## Classification with k-nearest neighbours

Points that are close in feature space probably share a label.

A classifier maps a feature vector like \[5.1, 3.5, 1.4, 0.2\] to a label like *I. setosa*. Without ML you could use a giant dictionary or hand-written rules, which works for simple, well-separated data. ML helps when datasets are complex and relationships unclear.

### The algorithm

- **1-NN**: predict the label of the single closest training point.
- **k-NN**: find the k closest training points and take a majority vote. Slide example with k = 5: 3 yellow, 2 green → predict yellow.
- Distance in feature space stands for similarity. Default in scikit-learn is Minkowski with p=2, which is Euclidean distance:

\\\[ d(\\mathbf{x}, \\mathbf{x}') = \\sqrt{\\sum\_{j=1}^{p} (x_j - x'\_j)^2} \\\]

```
def get_label(target, dataset, k=5):
    distances = []
    for datapoint in dataset:
        distances.append(distance(target, datapoint))
    pairs = zip(distances, dataset)
    top_k = sorted(pairs)[:k]
    return most_common([neighbor.label for neighbor in top_k])
```

### Hyperparameters

Hyperparameters are model *settings* that you choose (by hand or algorithmically), not values the model learns. For kNN: n_neighbors=5, weights='uniform', metric='minkowski', p=2.

### Why scaling matters for kNN

If one feature spans 0–100,000 and another spans 0–1, the large one dominates the distance. After rescaling, both features count **equally** when measuring distance, and the nearest neighbour can change.

Is kNN parametric or non-parametric? What does that cost you?

Non-parametric: no fixed set of weights, it keeps the whole training set. It needs more data, is less efficient at prediction time (distance to every point), and is flexible.

Small k vs large k: which overfits?

Small k (especially k = 1) follows individual noisy points, so it overfits; large k smooths toward the majority class and can underfit.

Lectures 3–4

## Linear regression & ordinary least squares

Predict a continuous value as a weighted sum of features, choosing weights that minimize squared error.

Running example: happiness *h* (Cantril ladder, 0–10) vs wealth *r* (GDP per capita), e.g. Ireland 7.02 / 84,340; Norway 7.55 / 64,341. Goal: find f with h = f(r). **Hypothesis**: the relationship is linear. That hypothesis is an *assumption about the model*.

model (β are the parameters, learned in training)\\\[ y' = \\beta_0 + \\beta_1 x_1 + \\beta_2 x_2 + \\dots + \\beta_p x_p \\\]

We write one such equation per instance *i*: \\( y'\_i = \\beta_0 + \\beta_1 x\_{i,1} + \\dots + \\beta_p x\_{i,p} \\). We choose the betas to minimize the squared vertical distances \\( \\sum d_i^2 \\), i.e. the mean squared error:

\\\[ \\text{MSE} = \\frac{1}{n}\\sum\_{i=1}^{n} (y_i - y'\_i)^2 \\\]

### Closed-form solution (OLS)

normal equation\\\[ \\boldsymbol\\beta = (X^\\top X)^{-1} X^\\top \\mathbf{y} \\\]

Derivation (shown, but *out of scope*): write \\( \\mathbf{y} = X\\boldsymbol\\beta + \\mathbf{e} \\), square the error \\( \\mathbf{e}^\\top\\mathbf{e} = \\mathbf{y}^\\top\\mathbf{y} - 2\\boldsymbol\\beta^\\top X^\\top \\mathbf{y} + \\boldsymbol\\beta^\\top X^\\top X \\boldsymbol\\beta \\), set the derivative \\( -2X^\\top \\mathbf{y} + 2X^\\top X\\boldsymbol\\beta = 0 \\), solve for β.

In-class calculation matrix dimensions

p = number of features (counting the column of ones), n = number of instances. Knowing matrix multiplication is important.

| Matrix | Shape | Matrix | Shape |
| --- | --- | --- | --- |
| \\(X\\) | n × p | \\((X^\\top X)^{-1}X^\\top\\) | p × n |
| \\(X^\\top\\) | p × n | \\(\\mathbf{y}\\) | n × 1 |
| \\(X^\\top X\\) | p × p | \\(\\boldsymbol\\beta = (X^\\top X)^{-1}X^\\top \\mathbf{y}\\) | p × 1 |
| \\((X^\\top X)^{-1}\\) | p × p | → one weight (parameter) per feature |  |

### In Python

```
# add a column of 1s so beta_0 (the intercept) is part of beta
X = np.column_stack((np.ones(len(x)), x))
Y = np.column_stack(y_train).T                      # n x 1
beta = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(Y)    # closed form

def predict(X_test, beta):
    Y_hat = X_test * beta.T          # elementwise: each feature times its weight
    return np.sum(Y_hat, axis=1)     # sum across a row = one prediction per instance
```

Why the column of 1s?

\\(\\beta_0\\) multiplies a feature that is always 1, so the intercept is learned just like every other weight. And sum(axis=1) adds up each row's weighted features; it does *not* sum across instances.

### Polynomial regression

\\\[ y' = \\beta_0 + \\beta_1 x_1 + \\beta_2 x_1^2 + \\beta_3 x_1^3 + \\dots + \\beta_d x_1^d \\\]

Still *linear in the betas*, so you solve it the same way, just with a bigger X: columns \\(1, x, x^2, \\dots, x^d\\), which gives d + 1 columns for degree d (one row per instance; the slide writes the shape as \\(p \\times (d+1)\\)). With more than one feature you also get powers of each feature. Code: np.column_stack(\[x, x\*\*2, x\*\*3\]). Higher degrees fit the happiness data more tightly, and eventually too tightly (overfitting).

### Three distinctions to know

|  | Parametric (e.g. linear regression) | Non-parametric (e.g. kNN) |
| --- | --- | --- |
| Efficiency | More efficient, needs less data | Less efficient, needs more data |
| Interpretability | More interpretable | Less interpretable |
| Flexibility | Less flexible, makes assumptions | More flexible (more likely to overfit), "no assumptions" |

- **Prediction**: how can we use X to predict Y? (How many people will fall asleep in class, given the words the professor uses.) **Inference**: what is the nature of the relationship between X and Y? (Does saying "GDP" cause students to fall asleep?)
- **Parameters** are weights the model learns. **Hyperparameters** are settings that control the model (k = 5, polynomial degree, learning rate).
- Closed-form solutions become impractical with **large data, many features, or more complex models**. That's why we need gradient descent.

n = 200 houses, 4 features plus an intercept column. Shapes of X, XᵀX and β?

p = 5, so X is 200 × 5, XᵀX is 5 × 5, β is 5 × 1.

Is \\(y = \\beta_0 + \\beta_1 x + \\beta_2 x^2\\) a linear model?

Yes, in the sense that matters for OLS: it's linear in the parameters β. The curve in x is non-linear, but you just add an \\(x^2\\) column to X.

Lectures 4–5

## Loss functions & gradient descent

Compute how the loss changes with each weight, then step every weight a little downhill. Repeat.

A **loss function** (or cost) measures the difference between the model's predictions and the true labels. MSE is the loss for regression. Training means minimizing it.

The derivative \\( f'(x) = dy/dx \\) is the slope. Follow the slope downhill to reach a minimum. A function can have **local minima** as well as the **global minimum**. With p features the loss lives in p dimensions, so we need methods that work in any dimension.

### Derivative of MSE (one feature)

\\\[ \\text{MSE} = \\frac{1}{n}\\sum\_{i=1}^{n}\\big(y_i - (\\beta_0 + \\beta_1 x_i)\\big)^2 \\\]

\\\[ \\begin{aligned}\\frac{\\partial \\text{MSE}}{\\partial \\beta_0} &= \\frac{1}{n}\\sum 2(y_i - \\beta_0 - \\beta_1 x_i)(-1)\\\\ &= -\\frac{2}{n}\\sum\_{i=1}^n (y_i - \\beta_0 - \\beta_1 x_i)\\end{aligned} \\\]

\\\[ \\begin{aligned}\\frac{\\partial \\text{MSE}}{\\partial \\beta_1} &= \\frac{1}{n}\\sum 2(y_i - \\beta_0 - \\beta_1 x_i)(-x_i)\\\\ &= -\\frac{2}{n}\\sum\_{i=1}^n x_i(y_i - \\beta_0 - \\beta_1 x_i)\\end{aligned} \\\]

The pattern: *(−2/n) × Σ (that weight's input) × (residual)*. For \\(\\beta_0\\) the input is the constant 1. With two features you get a third partial with \\(x\_{2i}\\) in front.

### Vector form

The residual \\( y_i - \\beta_0 - \\beta_1x\_{1i} - \\beta_2x\_{2i} = y_i - X_i\\boldsymbol\\beta \\) (with \\(X_i\\) a 1×3 row, β 3×1). Each partial is \\( -\\tfrac{2}{n} X\_{\\cdot j}^\\top(\\mathbf{y} - X\\boldsymbol\\beta) \\), a scalar. Stack them:

gradient: vector of partials, same shape as β (p × 1)\\\[ \\nabla\_{\\boldsymbol\\beta}\\text{MSE} = -\\frac{2}{n} X^\\top(\\mathbf{y} - X\\boldsymbol\\beta) \\\]

It tells you the shape of the loss surface and the direction of the error.

### The update rule

\\\[ \\boldsymbol\\beta \\leftarrow \\boldsymbol\\beta - \\eta \\, \\nabla\_{\\boldsymbol\\beta}\\text{MSE} \\\]

β is a parameter; **η (eta), the learning rate, is a hyperparameter**. We push the weights in the right direction and try again. If η is too large you overshoot and can diverge; too small and training crawls.

```
beta = np.random.randn(2, 1)                        # random init: "No problem!"
gradients = 2/n * X.T.dot(X.dot(beta) - Y)          # same as -(2/n) X^T (y - X beta)
beta = beta - alpha * gradients
```

Sign check

The code writes \\(X\\beta - y\\) with a plus sign out front; the math writes \\(y - X\\beta\\) with a minus. Same thing. Don't lose a sign on the exam.

Why does every partial derivative have a factor of the feature value \\(x\_{ji}\\) in it?

Chain rule: the inner function \\(y_i - \\sum_j \\beta_j x\_{ji}\\) has derivative \\(-x\_{ji}\\) with respect to \\(\\beta_j\\). A weight attached to a large input has more effect on the loss.

When do we stop updating?

After a set number of epochs, or when the validation loss stops improving for some number of epochs (patience). See the next unit.

Lectures 4–6

## Generalization, model selection & bias–variance

A model is only as good as its performance on data it never trained on.

### Splits

- **Training data**: learn the model's parameters.
- **Validation / development data**: tune hyperparameters and compare models.
- **Test data**: similar distribution, different instances. Pretend it's the real world. Tuning on the test set gives an unrealistically optimistic estimate.

### k-fold cross-validation

Not much data to spare? Split it into K equal parts. Train on K − 1 parts and test on the remaining part; rotate so every part is the test fold once; **average the metric across folds**. With very little data set K = n: **leave-one-out** cross-validation, which tests on every point.

### Hyperparameter search

```
# Grid search: every combination
for lr in np.linspace(0.01, 0.1, num=5):
    for pd in np.linspace(4, 10, num=4):
        ...  # train, eval on validation

# Random search: sample combinations
for i in range(20):
    lr = np.random.uniform(0.01, 0.1)
    pd = np.random.randint(4, 10)
    ...
```

The slides call random search **strictly better**: it samples each hyperparameter at more distinct values and doesn't duplicate parameter values.

### Recipe for evaluating a classifier

1. Split into train / validation / test.
2. Train a first model on training data.
3. Evaluate on training data.
4. Evaluate on validation data.
5. Try different models and hyperparameters.
6. Take the best setup and evaluate it *once* on the test data.

Lecture 7's picture: three configs (LR = 0.01/PD = 2, LR = 0.03/PD = 7, LR = 0.02/PD = 5) get validation F1 of 0.65, 0.72, 0.44, so you run LR = 0.03, PD = 7 on the test set, with the number of epochs set by the stopping rule.

### Over- and underfitting

- An **epoch** is one pass through the whole training set. As you train longer, training loss keeps falling; validation loss falls, then rises.
- **Overfitting**: fitting the training data too exactly at the cost of generalization. On the MSE-vs-polynomial-degree plot: low training MSE, high validation MSE at high degree.
- **Underfitting**: fitting too loosely, so performance is poor on training *and* test (low degree).
- **Patience** (early stopping): stop when the model hasn't improved on validation for X epochs.

### Bias–variance trade-off

| Error source | Meaning | Shows up as |
| --- | --- | --- |
| **Bias** | Wrong assumptions about the model | Underfitting; darts clustered off-centre |
| **Variance** | Sensitivity to the particular training data | Overfitting; darts scattered |
| **Noise** | Irreducible, random | Can't be fixed |

It's hard to lower bias and variance at once. Regularization deliberately trades a little more bias for lower variance.

200 examples, 5-fold CV. How many models are trained, and on how many examples each?

5 models, each trained on 160 and tested on 40. Leave-one-out would train 200 models on 199 examples each.

A degree-15 polynomial has training MSE 0.1 and validation MSE 9.0. Diagnosis?

Overfitting (high variance). Lower the degree, regularize, or get more data.

Lectures 5–6

## Logistic regression

Squash a linear score through a sigmoid to get a probability; train by minimizing cross-entropy.

**Notation change:** weights are now *w* (not β), the linear score is *z*, the bias is *b*, and all parameters together are θ = {w, b}.

sigmoid: maps any score to (0, 1)\\\[ \\sigma(z) = \\frac{1}{1 + e^{-z}}, \\quad z = \\mathbf{w}\\cdot\\mathbf{x} + b \\\]

class probabilities\\\[ \\begin{aligned} P(y{=}1\\mid x) &= \\sigma(\\mathbf{w}\\cdot\\mathbf{x}+b)\\\\ P(y{=}0\\mid x) &= 1 - \\sigma(\\mathbf{w}\\cdot\\mathbf{x}+b)\\end{aligned} \\\]

Predict class 1 when \\(\\hat y > 0.5\\) (that is, when z > 0). Useful values: σ(0) = 0.5, and \\( \\sigma'(z) = \\sigma(z)(1-\\sigma(z)) \\), which is why it's "nicely differentiable."

### Cross-entropy loss, built up

1. Probability of the correct label in one expression (y is 0 or 1): \\( p(y\\mid x) = \\hat y^{\\,y}(1-\\hat y)^{1-y} \\)
2. Take logs (products become sums; easier to compute and differentiate): \\( \\log p(y\\mid x) = y\\log\\hat y + (1-y)\\log(1-\\hat y) \\)
3. That's the (log-)likelihood we want to *maximize*. Losses are *minimized*, so negate it:

binary cross-entropy (log = natural log)\\\[ L\_{CE}(\\hat y, y) = -\\big\[\\,y\\log\\hat y + (1-y)\\log(1-\\hat y)\\,\\big\] \\\]

training objective: average over the m training points\\\[ \\hat\\theta = \\arg\\min\_\\theta \\frac{1}{m}\\sum\_{i=1}^m L\_{CE}\\big(f(x^{(i)};\\theta),\\, y^{(i)}\\big) \\\]

### The gradient (memorize the end result)

\\\[ \\frac{\\partial L\_{CE}}{\\partial w_j} = -(y - \\hat y)\\,x_j = (\\hat y - y)\\,x_j \\\]

\\\[ \\frac{\\partial L\_{CE}}{\\partial b} = -(y - \\hat y) = \\hat y - y \\\]

In words: *(estimate − true label) × the input for that weight*. Derivation (not needed by hand; autodiff tools do it): distribute the derivative, use \\( \\tfrac{d}{dx}\\ln x = 1/x \\), the chain rule, and \\( \\sigma' = \\sigma(1-\\sigma) \\); terms cancel to \\( \[\\sigma(\\mathbf{w}\\cdot\\mathbf{x}+b) - y\]\\,x_j \\).

update for all parameters at once\\\[ \\theta^{t+1} = \\theta^{t} - \\eta\\,\\nabla L(f(x;\\theta), y), \\qquad \\nabla L = \\begin{bmatrix} \\partial L/\\partial w_1 \\\\ \\vdots \\\\ \\partial L/\\partial w_n \\\\ \\partial L/\\partial b\\end{bmatrix} \\\]

In-class calculation SGD step 1

y = 1, x = \[3, 2\], w = \[0, 0\], b = 0, η = 0.1 (will solar activity damage this transformer?)

1. Score: z = 0·3 + 0·2 + 0 = 0, so \\(\\hat y = \\sigma(0) = 0.5\\).
2. Error term: \\(\\hat y - y = 0.5 - 1 = -0.5\\).
3. Gradient: \[−0.5·3, −0.5·2, −0.5\] = **\[−1.5, −1.0, −0.5\]** for (w₁, w₂, b).
4. Update: θ = \[0, 0, 0\] − 0.1·\[−1.5, −1.0, −0.5\] = **\[0.15, 0.1, 0.05\]**.

In-class calculation SGD step 2

Next example, starting from step 1's weights: y = 0, x = \[1, 0\], w = \[0.15, 0.1\], b = 0.05, η = 0.1.

1. z = 0.15·1 + 0.1·0 + 0.05 = 0.2, so \\(\\hat y = \\sigma(0.2) = 1/(1+e^{-0.2}) = 0.5498\\).
2. Error term: 0.5498 − 0 = 0.5498.
3. Gradient: \[0.5498·1, 0.5498·0, 0.5498\] = **\[0.5498, 0, 0.5498\]**.
4. Update: \[0.15 − 0.05498, 0.1 − 0, 0.05 − 0.05498\] = **\[0.09502, 0.1, −0.00498\]**.

Note that w₂ didn't move: its input was 0, so it had no influence on this prediction.

Calculator

You'll need \\(e^{x}\\) for the sigmoid and natural log (ln) for cross-entropy. The loss of step 2's example is \\(-\\ln(1-0.5498) = 0.798\\).

If y = 1 and the model says ŷ = 0.99, roughly how big are the gradient and the loss?

Gradient ∝ (0.99 − 1) = −0.01, tiny. Loss = −ln(0.99) ≈ 0.01. Confident and correct means almost no update.

Why use the log-likelihood instead of the likelihood?

Products of many probabilities become sums of logs: numerically stable and much easier to differentiate. Maximizing log-likelihood = minimizing negative log-likelihood = cross-entropy.

Lectures 6–7

## Stochastic gradient descent & convexity

You don't need every data point to estimate which way is downhill.

Full-batch gradient descent averages the gradient over the whole dataset before each step; as the dataset grows, each update gets slower. **SGD** updates on one point at a time.

|  | Stochastic (1 point) | Mini-batch | Full batch |
| --- | --- | --- | --- |
| Points per update | 1 | e.g. 16, 32, 64 | whole training set (e.g. 12,345) |
| Updates per epoch | training-set size | size ÷ batch size | 1 |
| Speed | Fast updates | Balance | Slow updates |
| Gradient | Noisy | Balance of stability and speed | Smooth (average) |

In everyday usage "SGD" usually means mini-batch SGD, and people drop "mini": "what's the batch size?"

### Convexity

A function is **convex** if the line between any two points on the curve lies on or above the curve. Then any minimum is the global minimum. **Cross-entropy loss for logistic regression is convex.** For non-convex functions, the randomness in SGD can help break out of local minima.

### SGD in code (logistic regression)

```
weights = np.random.uniform(size=3)              # 1. random init
bias = np.random.uniform()
y_pred = sigmoid(np.dot(X, weights) + bias)      # 2. estimate y

def compute_loss(y_true, y_pred):                # 3. binary cross-entropy
    return -np.mean(y_true*np.log(y_pred) + (1-y_true)*np.log(1-y_pred))

def compute_gradients(X, y_true, y_pred):        # 4. gradient (mean over the batch)
    grad_b = np.mean(y_pred - y_true)
    grad_w = np.matmul(X.T, y_pred - y_true) / len(y_true)
    return grad_w, grad_b

weights -= alpha * grad_w                        # 5. update (alpha = learning rate)
bias    -= alpha * grad_b

def predict(X):                                  # after training
    y_hat = sigmoid(np.dot(X, weights) + bias)
    return [1 if i > 0.5 else 0 for i in y_hat]
```

Lightly cleaned up from the slide code (same five steps). Lecture 11 adds: shuffle the data each epoch, which can improve convergence.

### GPUs

GPUs were built for graphics, which is matrix operations on images. Parallelizing matrix operations makes them far faster, and SGD is matrix operations too. That's why GPUs are central to ML (GeForce 256 in 1999 at $250 vs H100 in 2022 at $30–50k). The department has GPU servers (ssh MACID@server.cas.mcmaster.ca); check watch -n 5 nvidia-smi and don't run jobs on GPUs already in use. Large data centres can use about 1 million litres of water per day.

Training set of 1,000 points, batch size 50. Updates per epoch?

1000 / 50 = 20 updates per epoch. Pure SGD would make 1,000; full batch would make 1.

Lectures 7–8

## Multinomial logistic regression (softmax)

One score per class, turned into a probability distribution with softmax.

More than two classes (mushroom spore prints: paddy straw, puffball, deathcap, purple jellydisc, destroying angel). Represent the label as a **one-hot vector** of length K: class 2 of 5 is y = \[0 0 1 0 0\] (classes counted from 0).

Instead of one score z, output K scores \\(\[z_1, \\dots, z_K\]\\), with \\( z_i = \\mathbf{w}\_i\\cdot\\mathbf{x} + b_i \\). The weights become a K × f matrix W; each row is like a prototype of its class.

softmax: a generalization of the sigmoid\\\[ \\text{softmax}(z_i) = \\frac{\\exp(z_i)}{\\sum\_{j=1}^{K}\\exp(z_j)}, \\quad 1 \\le i \\le K \\\]

Every output is positive and they sum to 1, e.g. 70% deathcap, 20% paddy straw, 10% puffball.

|  | Binary logistic | Multinomial logistic |
| --- | --- | --- |
| Output | scalar ŷ via sigmoid | vector of K probabilities via softmax |
| Weights | vector w, \[1 × f\] | matrix W, \[K × f\] |
| Label | y ∈ {0, 1} | one-hot vector |

Why "generalization of the sigmoid"

With two classes and scores (z, 0): \\( e^{z}/(e^{z}+e^{0}) = 1/(1+e^{-z}) = \\sigma(z) \\).

Scores \[1, 1, 1\]. What does softmax give?

\[1/3, 1/3, 1/3\]. Equal scores give equal probabilities; only differences between scores matter.

Lectures 7–8

## Evaluating classifiers

Accuracy alone can hide a useless or dangerous model. Compare against baselines and look per class.

\\\[ \\text{Precision} = \\frac{TP}{TP+FP} \\qquad \\text{Recall} = \\frac{TP}{TP+FN} \\qquad \\text{Accuracy} = \\frac{TP+TN}{TP+TN+FP+FN} \\\]

\\\[ F_1 = 2\\cdot\\frac{\\text{Precision}\\cdot\\text{Recall}}{\\text{Precision}+\\text{Recall}} \\quad\\text{(harmonic mean)} \\\]

**Precision**: of the things I bought, how many were on the list? (Of predicted positives, how many are right.) **Recall**: of the things on my list, how many did I buy? (Of actual positives, how many did I find.) **Macro** precision/recall = the plain average of the per-class values.

In-class calculation blue vs green

10 items: 7 blue, 3 green. The classifier's "blue" box holds 5 blue + 1 green; its "green" box holds 2 green + 2 blue.

|  | Predicted blue | Predicted green | Total |
| --- | --- | --- | --- |
| Actually blue | 5 | 2 | 7 |
| Actually green | 1 | 2 | 3 |
| Total | 6 | 4 | 10 |

1. Accuracy = (5 + 2) / 10 = **70%**.
2. Precision(green) = 2/4 = 0.5; precision(blue) = 5/6. Macro precision = (0.5 + 0.833)/2 = **0.67**.
3. Recall(blue) = 5/7 = 0.71; recall(green) = 2/3. Macro recall = (0.714 + 0.667)/2 = **0.69**.

### Simple baselines

- **Majority baseline**: always guess the most common class. If 80% of data is blue, that scores 80%, so our 70% classifier is not very good.
- **Random baseline**: guess uniformly at random. With 2 classes: 50%.

In-class calculation F1 table

| Precision | Recall | F1 |
| --- | --- | --- |
| 0.5 | 0.5 | 0.5 |
| 0.7 | 0.2 | 2(0.14)/0.9 = 0.311 |
| 0.4 | 0.5 | 2(0.20)/0.9 = 0.444 |

The harmonic mean is dragged toward the smaller value, so you can't hide a bad recall behind a great precision.

### The confusion matrix

Rows = true label, columns = predicted label. It shows how often one class is "confused" with another. The best possible matrix has everything on the diagonal.

| True ↓ / Predicted → | Paddy straw | Puffball | Deathcap |
| --- | --- | --- | --- |
| Paddy straw | 50 | 2 | 3 |
| Puffball | 102 | 19 | 5 |
| Deathcap | 71 | 40 | 0 |

This one is bad: it calls most things paddy straw and never finds a single deathcap (practice problem 5 works through it).

"It gets 95% accuracy!"

Would you eat a mushroom based on that? Ask: what's the majority baseline? What's the recall on deathcaps? Where did the data come from? A rare, deadly class can be missed entirely while accuracy looks great.

Spam filter: which metric matters more if legit email in the spam folder is very costly?

Precision for the spam class: of the emails flagged as spam, how many really are spam. False positives are the costly error.

Lectures 8–9

## Feature engineering & regularization

Penalize large weights so the model can't contort itself to fit noise.

### Feature engineering

The process of turning raw data into features a model can use. Examples from class: identifying visible actions in lifestyle vlogs (Prof. Oana Ignat), and predicting deception from trial videos using guilty / not-guilty / exoneration verdicts (Innocence Project) as labels.

Lots of features (say 10,000) cause problems:

- More features to overfit to.
- Irrelevant features make the true signal harder to find.
- Feature vectors become sparse.
- In high-dimensional space all points tend to be a similar distance apart, which is bad for kNN.

### Ridge (L2) and lasso (L1)

residual sum of squares (unaveraged MSE)\\\[ \\text{RSS} = \\sum\_{i=1}^{n}\\Big(y_i - \\beta_0 - \\sum\_{j=1}^{p}\\beta_j x\_{ij}\\Big)^2 \\\]

ridge = L2\\\[ \\text{RSS} + \\lambda\\sum\_{j=1}^{p}\\beta_j^2 \\\]

LASSO = L1\\\[ \\text{RSS} + \\lambda\\sum\_{j=1}^{p}|\\beta_j| \\\]

- λ is the **strength of regularization** (a hyperparameter; tune it on validation data). The added sum is the **shrinkage penalty**. The intercept β₀ isn't penalized (the sum starts at j = 1).
- λ = 0 gives ordinary least squares; as **λ → ∞, all parameters → 0**. In between (credit-card-debt example with Income, Limit, Rating, Student): a good range where useless variables are near zero and a few keep non-zero coefficients.
- Regularizing trades **lower variance for higher bias**.

### Why L1 zeros things out and L2 doesn't

L2 gradient term: proportional to the weight\\\[ \\frac{\\partial}{\\partial\\beta_j}\\lambda\\beta_j^2 = 2\\lambda\\beta_j \\\]

L1 gradient term: constant size\\\[ \\frac{\\partial}{\\partial\\beta_j}\\lambda|\\beta_j| = \\lambda\\,\\text{sign}(\\beta_j) \\\]

L2's push shrinks as the weight approaches 0, so weights get small but rarely exactly 0. L1 pushes with the same force no matter how small the weight, so weights near zero get forced to zero. In the coefficient-path plots, L1 features drop to zero one by one at distinctly different λ values: **feature selection**.

| L1 / Lasso | L2 / Ridge |
| --- | --- |
| Linear penalty | Quadratic penalty |
| "Selects" features, more interpretable | Retains features, less interpretable |
| Less stable with correlated features | Shrinks but does not eliminate correlated features |

Correlated features example (house prices from total sq ft, living-area sq ft, bedrooms, year built): one L1 run may drop total sq ft, the next may drop living-area sq ft. L2 keeps both, shrunk.

### Same idea for classifiers (likelihood form)

\\\[ \\hat\\theta = \\arg\\max\_\\theta \\Big\[\\sum\_{i=1}^m \\log P(y^{(i)}\\mid x^{(i)})\\Big\] - \\alpha\\sum\_{j=1}^n \\theta_j^2 \\quad\\text{(L2)} \\qquad \\ldots - \\alpha\\sum\_{j=1}^n |\\theta_j| \\quad\\text{(L1)} \\\]

Maximizing likelihood, so the penalty is *subtracted*; α is the regularization strength. In the TensorFlow Playground demo, logistic regression is a 1-layer neural network, and L1 switches off the input features it doesn't need.

You want a model that tells you which 5 of 200 survey questions matter. L1 or L2?

L1 (lasso): it drives irrelevant coefficients exactly to zero.

λ is set too high. What happens to training error and to bias?

All weights shrink toward 0, so the model underfits: training error rises and bias is high.

Lecture 10

## Support vector machines

Of all the boundaries that separate the classes, pick the one with the widest margin.

### Separating hyperplanes

- In a p-dimensional space, a **hyperplane is (p − 1)-dimensional** (a line in 2-D, a plane in 3-D): \\( \\beta_0 + \\beta_1x\_{i1} + \\dots + \\beta_px\_{ip} = 0 \\).
- Labels are **±1** (not 0/1). Classify by which side a point falls on: the expression is > 0 if yᵢ = +1 and \< 0 if yᵢ = −1.
- Many hyperplanes may separate the data. One too close to a class misclassifies new points. We want the **maximal margin classifier**: the margin is the distance from the boundary to the nearest points.
- Margin planes: H₁: \\(\\beta_0+\\sum\\beta_jx\_{ij} = 1\\) and H₂: \\(\\dots = -1\\). The ±1 is an arbitrary constant saying the planes are equidistant from the middle (any scale works; 1 is easiest). Margin = d⁺ + d⁻.
- **Support vectors** are the points on or near the margin. Moving other points changes nothing; moving a support vector moves the boundary. We only optimize for points near the boundary.

### Hard margin

\\\[ \\max\_{\\beta_0,\\dots,\\beta_p,\\,M} M \\quad\\text{s.t.}\\quad \\sum\_{j=1}^p\\beta_j^2 = 1,\\quad y_i(\\beta_0 + \\beta_1x\_{i1} + \\dots + \\beta_px\_{ip}) \\ge M \\;\\;\\forall i \\\]

argmax: find the parameters that maximize M. The product \\(y_i(\\cdot)\\) is positive exactly when the point is on the correct side, so ≥ M means every point is outside the margin. The norm constraint stops you from inflating M just by scaling the β's. **Fails if the data aren't separable.**

### Soft margin (support vector classifier)

\\\[ \\max M \\quad\\text{s.t.}\\quad \\sum_j\\beta_j^2 = 1,\\quad y_i(\\beta_0 + \\textstyle\\sum_j\\beta_jx\_{ij}) \\ge M(1-\\epsilon_i),\\quad \\epsilon_i \\ge 0,\\quad \\sum\_{i=1}^n \\epsilon_i \\le C \\\]

- More robust to individual points; classifies *most* points well.
- Each point gets a slack εᵢ: 0 = correct side of the margin; between 0 and 1 = inside the margin but correct side of the hyperplane; > 1 = misclassified.
- Here (ISL book) **C is a budget**: total violation allowed. Large C → more tolerance → wider margin. Decreasing C approaches the hard margin. This is the "primal form."

### Hinge loss form

ISL book\\\[ \\min\_{\\beta}\\Big\\{\\sum\_{i=1}^n \\max\\big\[0,\\,1 - y_i(\\beta_0 + \\beta_1x\_{i1} + \\dots + \\beta_px\_{ip})\\big\] + \\lambda\\sum\_{j=1}^p\\beta_j^2\\Big\\} \\\]

MML book (and scikit-learn)\\\[ \\min\_{b,\\mathbf{w}}\\Big\\{C\\sum\_{i=1}^n \\max\\big\[0,\\,1 - y_i(b + \\mathbf{w}\\cdot\\mathbf{x}\_i)\\big\] + \\tfrac12\\|\\mathbf{w}\\|^2\\Big\\} \\\]

- The hinge term is 0 when \\(y_if(x_i) \\ge 1\\) (correct and past the margin). It is positive in the middle region (correct, but not far enough past the margin to satisfy us) and grows linearly for misclassified points.
- In the ISL form λ acts like the budget C from before (larger λ → wider margin).
- **In the MML form C has the opposite effect**: large C = high cost for errors = narrower margin, fewer violations. scikit-learn's SVC(C=…) uses this meaning.
- Either way you're trading off *margin width* against *how wrong misclassified points are allowed to be*.

In-class calculation checking boundaries

1-D data: A (x = −2, y = −1), B (−1, −1), C (1, +1), D (2, +1). Soft margin with C = 1:

\\\[ \\min\_{w,b,\\xi}\\ \\tfrac12 w^2 + C\\sum\_{i=1}^4 \\xi_i \\quad\\text{s.t.}\\quad y_i(wx_i+b) \\ge 1-\\xi_i,\\ \\ \\xi_i \\ge 0, \\qquad \\xi_i = \\max(0,\\,1 - y_i(wx_i+b)) \\\]

1. **Constraints**: A: −1(−2w + b) ≥ 1 − ξ_A → 2w − b ≥ 1 − ξ_A. B: w − b ≥ 1 − ξ_B. C: w + b ≥ 1 − ξ_C. D: 2w + b ≥ 1 − ξ_D.
2. **Slacks at w = b = 0.5**: A: 2(0.5) − 0.5 = 0.5 → ξ_A = 0.5. B: 0.5 − 0.5 = 0 → ξ_B = 1.0. C: 1 → ξ_C = 0. D: 1.5 → ξ_D = 0.
3. **Objective**: ½(0.5)² + 1·(0.5 + 1 + 0 + 0) = 0.125 + 1.5 = **1.625**.
4. **Optimal?** No. With **w = 1, b = 0** every constraint holds with ξ = 0 (B and C sit exactly on the margin, so they are the support vectors), and the objective is ½(1)² = **0.5**.

Why can't we just make w huge so every \\(y_i(wx_i+b)\\) is way above 1?

The ½‖w‖² term (or the norm constraint in the hard-margin form) penalizes large w. A larger w means a narrower margin.

A point has slack ξ = 1.4. Where is it?

On the wrong side of the hyperplane (misclassified), since \\(y_if(x) = 1 - 1.4 = -0.4 \\lt 0\\).

Lecture 11

## Training SVMs & the kernel trick

Train with SGD on the hinge loss; get non-linear boundaries by swapping the dot product for a similarity function.

### SGD for the SVM

loss, with the bias folded into w as with OLS\\\[ \\ell = C\\sum\_{i=1}^n\\max\\big\[0,\\,1 - y_i(\\mathbf{w}\\cdot\\mathbf{x}\_i)\\big\] + \\tfrac12\\|\\mathbf{w}\\|^2 \\\]

The hinge isn't differentiable at the kink (left and right gradients differ). That's fine; pick one side and carry on. Per-point gradient, two cases depending on the max:

\\\[ \\frac{\\partial\\ell}{\\partial\\mathbf{w}} = \\begin{cases}\\mathbf{w}, & 1 - y_i(\\mathbf{w}\\cdot\\mathbf{x}\_i) \\le 0 \\quad\\text{(beyond the margin)}\\\\ \\mathbf{w} - C\\,y_i\\mathbf{x}\_i, & 1 - y_i(\\mathbf{w}\\cdot\\mathbf{x}\_i) \\gt 0 \\quad\\text{(inside margin or wrong)}\\end{cases} \\\]

The \\(\\mathbf{w}\\) comes from ½‖w‖²; the \\(-Cy_i\\mathbf{x}\_i\\) from the hinge term. Steps: (1) randomly initialize weights, (2) compute the loss gradient, (3) update w = w − learning_rate · gradient, (4) repeat for E epochs, shuffling the data each epoch.

### Non-linear boundaries

You could add quadratic features (\\(\\beta\_{j1}x\_{ij} + \\beta\_{j2}x\_{ij}^2\\)), but every extra feature adds sums to the constraints and gets complicated fast. Instead, use the **kernel trick**: reformulate the SVM with a similarity function between two observations.

| Kernel | Formula | Notes |
| --- | --- | --- |
| Linear | \\( K(x_i, x\_{i'}) = \\sum\_{j=1}^p x\_{ij}x\_{i'j} \\) | Just the inner (dot) product |
| Polynomial | \\( K(x_i, x\_{i'}) = \\big(1 + \\sum\_{j=1}^p x\_{ij}x\_{i'j}\\big)^d \\) | Degree d (cubic boundary when d = 3) |
| Radial basis (RBF) | \\( K(x, x') = \\exp\\big(-\\gamma\\\|x - x'\\\|^2\\big) \\) | γ scales similarity. Lower γ → fuzzier, smoother boundary, more global influence. Higher γ → more local, wigglier. |

decision function: f(x) > 0 → one class, f(x) \< 0 → the other\\\[ f(x) = \\beta_0 + \\sum\_{i=1}^{n}\\alpha_i K(x, x_i) \\\]

The αᵢ and β₀ are estimated in training. Fitting involves inner products between all n(n − 1)/2 pairs of points, which sounds like a lot, but **most αᵢ are 0**: only support vectors contribute.

### Dual form (not derived; see 4ML3 / MML book 12.3)

\\\[ \\min\_{\\boldsymbol\\alpha}\\ \\frac12\\sum\_{i=1}^N\\sum\_{j=1}^N y_iy_j\\alpha_i\\alpha_j\\langle\\mathbf{x}\_i,\\mathbf{x}\_j\\rangle - \\sum\_{i=1}^N\\alpha_i \\quad\\text{s.t.}\\quad \\sum\_{i=1}^N y_i\\alpha_i = 0,\\quad 0 \\le \\alpha_i \\le C \\\]

- The constraints now involve pairs of points.
- \\(\\sum y_i\\alpha_i = 0\\): the α's are balanced between the two classes. You need points from both for the boundary.
- \\(y_iy_j\\) is positive for same-label pairs and negative for different-label pairs, so the objective favours α on pairs of *different* labels that are *similar* (close in feature space): the points near the boundary.
- We push α up toward C, but each value depends on the others. **C caps the "importance" of any single point.**
- Replace \\(\\langle\\mathbf{x}\_i,\\mathbf{x}\_j\\rangle\\) with \\(K(\\mathbf{x}\_i,\\mathbf{x}\_j)\\): that's the kernel trick.

In practice this is solved with quadratic programming. Assignment 2 uses from sklearn.svm import SVC, where a higher C means a higher cost for being wrong (MML sense).

For a correctly classified point beyond the margin, what's the hinge part of its SGD gradient?

Zero. The gradient is just w from the regularizer, so the update only shrinks w slightly.

Your RBF SVM traces a tiny island around every training point and fails on validation. Raise or lower γ?

Lower γ, which makes similarity decay more slowly and gives each point broader, smoother influence. (Lowering C in sklearn also helps.)

Lecture 11

## Data preprocessing

Models need vectors of numbers. How you build them depends on what kind of data you have and what question you're asking.

### Four kinds of data

| Type | Meaning | Examples | Encode as |
| --- | --- | --- | --- |
| **Nominal** | Labels, categories, groups | species, gender identity, nationality | One-hot vector |
| **Ordinal** | Ordered rankings | class grade, Likert scale | Ordered numbers, e.g. −2 … +2 |
| **Interval** | Numeric, no true zero | date, time | Number |
| **Ratio** | Numeric, true zero | height, weight, income | Number |

5-point Likert: strongly disagree −2, disagree −1, neither 0, agree +1, strongly agree +2. The order makes sense, so numbers work. Five crab-like species (convergently evolved, not actually related) have no order, so use one-hot \[0 0 1 0 0\]; integers 1–5 would invent an order and distances that don't exist.

### Look at your data

- Seaborn scatterplot matrices show pairwise correlations for a small subset of features.
- Mean and standard deviation can be insightful but **don't show the full picture**. The "Datasaurus" sets all share X mean 54.26, Y mean 47.83, X SD 16.76, Y SD 26.93, correlation −0.06, yet one is a dinosaur. Anscombe's quartet: four very different datasets with the same summary statistics. Always plot.

### Preparation checklist

1. Convert categorical data to one-hot vectors.
2. Handle missing features: drop the instance, or keep it and add an UNKNOWN category if the feature is categorical.
3. Rescale features: subtract the mean, divide by the standard deviation. This gives features the same weight/scale and can improve convergence.
4. Remove outliers (thoughtfully, below).
5. Create your train/dev/test splits.

standardization (z-score)\\\[ x' = \\frac{x - \\mu}{\\sigma} \\\]

### Outliers: it depends

| Case | Decision |
| --- | --- |
| Drinking study: time to type texts. Short delays reflect motor control (impaired when drinking); long ones reflect thinking/deliberation. | **Remove** the long ones to study motor control. |
| Chimera opinion survey designed to take 10 min: some times are 0 (timer/JS failure), some 5+ hours (tab left open). | **Remove**: measurement errors. |
| Restaurant reviews with no words from the sentiment lexicon, e.g. "I would not go there again." | **Keep**: still clearly negative. The lexicon is the problem. |
| Non-normal / bimodal data (people with varying abilities, costs of goods). | Often **keep**: a smaller but important part of the distribution. |

Practical rule when you do remove: drop values more than **3 standard deviations** above the mean.

Encode "T-shirt size: S, M, L, XL" and "favourite colour: red, green, blue".

Size is ordinal, so map to 0, 1, 2, 3. Colour is nominal, so one-hot: red = \[1,0,0\], green = \[0,1,0\], blue = \[0,0,1\].

Lecture 12

## Decision trees & ensembles

Ask the question that removes the most uncertainty, then repeat on each half.

Nodes split the feature space into regions with decision rules; a series of decisions splits it many times. **Leaves** hold a label (classification) or a value (regression). 20Q is a big decision tree. Trees are **very interpretable**.

### Method

1. Start with a root node.
2. Choose a feature (by information gain or Gini index).
3. Split on it.
4. Repeat until: no further gain, all samples share a class, max depth reached, or min samples per node reached.

entropy: uncertainty in the labels (0 = pure)\\\[ H(S) = -\\sum_i p_i\\log_2 p_i \\\]

information gain of a split\\\[ IG = H(S) - \\sum\_{v}\\frac{|S_v|}{|S|}H(S_v) \\\]

Calculator

No log₂ button? Use \\( \\log_2 x = \\ln x / \\ln 2 \\). Handy values: H(½, ½) = 1, H(¾, ¼) = 0.8113, H(⅝, ⅜) = 0.9544.

In-class calculation Will you pass the test? (binary)

| ID | Attends class (AC) | ≥3 hrs studied (HS) | Has tutor (HT) | Pass |
| --- | --- | --- | --- | --- |
| 1 | 0 | 0 | 0 | 0 |
| 2 | 0 | 0 | 1 | 1 |
| 3 | 0 | 1 | 0 | 1 |
| 4 | 0 | 1 | 1 | 0 |
| 5 | 1 | 0 | 0 | 1 |
| 6 | 1 | 0 | 1 | 1 |
| 7 | 1 | 1 | 0 | 0 |
| 8 | 1 | 1 | 1 | 1 |

1. **Root entropy**: 5 pass, 3 fail. H = −(⅝ log₂ ⅝ + ⅜ log₂ ⅜) = 0.4237 + 0.5306 = **0.9544**.
2. **Try AC**: AC = 0 → labels {0,1,1,0}, H = 1. AC = 1 → {1,1,0,1}, H = 0.8113. End entropy = 4/8·1 + 4/8·0.811 = 0.906. Gain = 0.954 − 0.906 = **0.048**.
3. **Try HS**: HS = 0 → {0,1,1,1}, H = 0.811; HS = 1 → {1,0,0,1}, H = 1. Gain = 0.048. **HT** also gives 0.048. A three-way tie, so take AC.
4. **Left branch (AC = 0)**, start entropy 1: HS and HT split it equally, so pick HS; then HT splits each side to single points. Leaves (HS, HT) = (0,0) → 0, (0,1) → 1, (1,0) → 1, (1,1) → 0.
5. **Right branch (AC = 1)**: HS or HT both give one pure side (entropy 0) and one mixed side (entropy 1), so it doesn't matter. With HS: HS = 0 → leaf 1; HS = 1 → split on HT: 0 → 0, 1 → 1.

### Continuous features

Sort the values and try a threshold at each unique value ("≥ t" vs "\< t"). Testing ≥ the smallest value is the same as not splitting.

In-class calculation hours studied as an integer

Same labels; HS is now hours: IDs 1–8 → 0, 1, 3, 4, 2, 1, 5, 3. AC and HT gains are unchanged (0.048).

1. **HS ≥ 1**: \<1 → {0}, H = 0. ≥1 → {1,1,0,1,1,0,1}, H = −(5/7 log₂ 5/7 + 2/7 log₂ 2/7) = 0.863. End = ⅛·0 + ⅞·0.863 = 0.755. Gain = **0.199**.
2. **HS ≥ 4**: \<4 → {0,1,1,1,1,1}, H = −(5/6 log₂ 5/6 + 1/6 log₂ 1/6) = 0.650. ≥4 → {0,0}, H = 0. Gain = 0.954 − 6/8·0.650 = **0.466**.
3. All thresholds: ≥0: 0, ≥1: 0.199, ≥2: 0.003, ≥3: 0.049, **≥4: 0.466**, ≥5: 0.199. Best root split: HS ≥ 4 → leaf 0.
4. Features can be reused: inside HS \< 4, split again on **HS ≥ 1**: \<1 → 0 (N = 1), ≥1 → 1 (N = 5). Done, and much simpler.

### Categorical features with many values

Most algorithms make binary splits. With AC ∈ {0 never, 1 sometimes, 2 always} you could try every grouping ({0,1} vs {2}, {0,2} vs {1}, {1,2} vs {0}), but that explodes as categories grow. Shortcut: **order categories by their mean label** (AC = 0 → ⅓, AC = 1 → ⅔, AC = 2 → 1), then only test splits along that order, like a continuous feature. Works for regression targets too (order by mean score).

### Regression trees

Follow the tree to a leaf and predict the **mean** of the training targets in that leaf (e.g. exam scores 40, 55, 75, …). Splits are chosen by reduction in RSS. Termination criteria keep the tree from getting very deep.

### Gini impurity

\\\[ G = 1 - \\sum\_{i} p_i^2 \\\]

In-class calculation Gini for HS ≥ 4

1. Root: G = 1 − (⅜² + ⅝²) = **0.46875**.
2. HS ≥ 4 → {0, 0}: G = 0. HS \< 4 → {0,1,1,1,1,1}: G = 1 − (⅚² + ⅙²) = 0.27778.
3. Weighted: 2/8·0 + 6/8·0.27778 = 0.2083. Reduction = 0.46875 − 0.2083 = **0.2604** (still compare against the other splits).

Trees cut the space into axis-aligned boxes by recursive binary splits. A partition that can't be produced by successive splits is impossible (ISL ch. 8, p. 335).

### Pruning

\\\[ \\sum\_{m=1}^{|T|}\\ \\sum\_{i:\\,x_i\\in R_m}\\big(y_i - \\hat y\_{R_m}\\big)^2 + \\alpha|T| \\\]

Grow a tree with your termination criteria, then tune α to minimize this. |T| = number of terminal nodes, Rₘ = the region at leaf m, \\(\\hat y\_{R_m}\\) = its mean prediction. Larger α → fewer leaves.

### Ensembles

| Method | How | Key point |
| --- | --- | --- |
| **Bagging** | Build B trees independently, each on a bootstrap sample (drawn from the training set *with replacement*); average: \\(\\hat f\_{avg}(x) = \\frac1B\\sum\_{b=1}^B \\hat f^{\\,b}(x)\\) | Better accuracy, less interpretable (too many trees) |
| **Feature importance** | For each feature, total RSS decrease / information gain at nodes that use it, averaged over all trees, rescaled | Gets some interpretability back |
| **Random forest** | Bagging, but at each split consider only a random subset of **√p** features | Decorrelates the trees, acts as a regularizer, tends to perform better |
| **Boosting** | Build trees *sequentially*; each new tree is trained on the mistakes (instances predicted incorrectly) of the forest so far | Keep looping, adding trees |
| **XGBoost** | Gradient boosting + engineering tricks: take the gradient of the loss, fit a new tree to predict the gradients, add it scaled by a learning rate | Uses the second derivative for more stable updates; often top-performing |

In-class example XGBoost gradients

A regression tree predicts 9, 7, 6 at its three leaves (left to right); true values are 10, 8, 1. For MSE, gradient = predicted − true = **\[−1, −1, 5\]**. Train a new tree to predict these gradients and combine it with the original prediction, scaled by the learning rate.

Sign detail: to *reduce* the loss you move against the gradient, \\( \\hat y \\leftarrow \\hat y - \\eta\\,g \\). Equivalently, fit the residuals (true − predicted) and add those. Practice problem 13 does one step.

Random forest with p = 25 features: how many features are considered at each split?

√25 = 5.

Bagging vs boosting in one line each?

Bagging: independent trees on bootstrap samples, averaged (reduces variance). Boosting: sequential trees, each correcting the previous trees' errors.

Midterm-style · new numbers

## Practice problems

Type answers to 3 decimal places (fractions like 5/7 work too) and press **Check**. Answers within about ±0.005 count. Work on paper first; the full solution is under each problem.

### 1 · One gradient-descent step (linear regression)

Units 3–4

Data: x = \[1, 2, 3\], y = \[2, 3, 5\]. Model \\(y' = \\beta_0 + \\beta_1x\\) with β₀ = 0, β₁ = 1, η = 0.1. Compute the MSE gradient and the updated weights.

∂MSE/∂β₀

∂MSE/∂β₁

new β₀

new β₁

Solution

Predictions 1, 2, 3 → residuals y − y' = 1, 1, 2 (MSE = 6/3 = 2).

∂/∂β₀ = −(2/3)(1 + 1 + 2) = −8/3 = −2.667. ∂/∂β₁ = −(2/3)(1·1 + 2·1 + 3·2) = −(2/3)(9) = −6.

β₀ = 0 − 0.1(−2.667) = 0.267; β₁ = 1 − 0.1(−6) = 1.6. Both weights went up, because the line was under the points.

### 2 · Logistic regression SGD step

Unit 6 · needs eˣ and ln

y = 1, x = \[1, 2\], w = \[0.5, −0.5\], b = 0, η = 0.5. Find ŷ, the cross-entropy loss, and the updated parameters.

ŷ

loss L_CE

new w₁

new w₂

new b

Solution

z = 0.5·1 + (−0.5)·2 + 0 = −0.5. ŷ = \\(1/(1 + e^{0.5})\\) = 1/(1 + 1.6487) = 0.3775.

Loss = −ln(0.3775) = 0.974 (y = 1, so only the first term survives).

ŷ − y = −0.6225. Gradient: w₁: −0.6225·1 = −0.6225; w₂: −0.6225·2 = −1.2449; b: −0.6225.

Update: w₁ = 0.5 + 0.5·0.6225 = 0.811; w₂ = −0.5 + 0.5·1.2449 = 0.122; b = 0 + 0.311 = 0.311.

### 3 · Softmax

Unit 8

Three class scores z = \[2, 1, 0\]. Compute the softmax probabilities.

p₁

p₂

p₃

Solution

e² = 7.389, e¹ = 2.718, e⁰ = 1. Sum = 11.107.

p = \[7.389, 2.718, 1\] / 11.107 = \[0.665, 0.245, 0.090\]. They sum to 1.

### 4 · Metrics from counts

Unit 9

100 test emails. TP = 40, FP = 10, FN = 20, TN = 30 (positive = spam). Compute precision, recall, F1, accuracy and the majority-class baseline accuracy.

precision

recall

F1

accuracy

majority baseline

Solution

P = 40/(40+10) = 0.8. R = 40/(40+20) = 0.667. F1 = 2(0.8)(0.667)/(1.467) = 0.727. Accuracy = (40+30)/100 = 0.70.

Actual spam = TP + FN = 60, actual not-spam = 40, so always guessing "spam" gets 0.60. The classifier beats it by 10 points.

### 5 · The mushroom confusion matrix

Unit 9 · uses the matrix from Lecture 8

Using the paddy straw / puffball / deathcap matrix above (rows = true), compute accuracy, precision and recall for paddy straw, and recall for deathcap.

accuracy

precision (paddy)

recall (paddy)

recall (deathcap)

Solution

Total = 55 + 126 + 111 = 292. Diagonal = 50 + 19 + 0 = 69. Accuracy = 69/292 = 0.236.

Precision(paddy) = 50 / column total (50 + 102 + 71 = 223) = 0.224. Recall(paddy) = 50 / row total 55 = 0.909.

Recall(deathcap) = 0/111 = 0. The model never identifies a deathcap: the worst possible failure for this task.

### 6 · Soft-margin slacks

Unit 11 · same setup as the in-class example

Points: A (x = −3, y = −1), B (−1, −1), C (0.5, +1), D (2, +1). Objective \\( \\tfrac12w^2 + C\\sum\\xi_i \\) with C = 2. Evaluate at w = 1, b = 0.

ξ_A

ξ_B

ξ_C

ξ_D

objective

Solution

ξᵢ = max(0, 1 − yᵢ(wxᵢ + b)). A: 1 − (−1)(−3) = −2 → 0. B: 1 − (−1)(−1) = 0 → 0. C: 1 − 0.5 = 0.5. D: 1 − 2 = −1 → 0.

Objective = ½(1)² + 2(0.5) = 0.5 + 1 = 1.5. C is correctly classified but inside the margin.

### 7 · Hinge-loss SGD update

Unit 12 · bias folded into w

w = \[1, −1\], C = 1, η = 0.1. Training point x = \[1, 1\], y = +1. Which case of the gradient applies? Compute the gradient and the updated w.

gradient \[1\]

gradient \[2\]

new w₁

new w₂

Solution

w·x = 1 − 1 = 0, so 1 − y(w·x) = 1 > 0: the point is inside the margin, use \\( \\mathbf{w} - Cy\\mathbf{x} \\).

Gradient = \[1, −1\] − 1·1·\[1, 1\] = \[0, −2\]. New w = \[1, −1\] − 0.1·\[0, −2\] = \[1, −0.8\].

Follow-up: for x = \[2, 0\], y = +1 we'd get w·x = 2 → beyond the margin → gradient = w, and w just shrinks a bit.

### 8 · Kernel values

Unit 12

x = \[1, 2\], x' = \[3, −1\]. Compute the linear kernel, the polynomial kernel with d = 2, and the RBF kernel with γ = 0.5.

linear

polynomial (d=2)

RBF (γ=0.5)

Solution

Dot product: 1·3 + 2·(−1) = 1. Polynomial: (1 + 1)² = 4.

‖x − x'‖² = (1−3)² + (2+1)² = 4 + 9 = 13. RBF = \\(e^{-0.5\\cdot 13} = e^{-6.5}\\) = 0.0015. These points are far apart, so their similarity is near 0.

### 9 · Entropy, information gain and Gini

Unit 14

A node has 6 examples with labels \[1, 1, 1, 0, 0, 0\]. Feature A sends {1, 1, 1, 0} left and {0, 0} right. Compute the root entropy, the information gain, and the Gini reduction.

H(root)

H(left)

info gain

Gini reduction

Solution

Root is 3/3, so H = 1. Left (¾, ¼): H = 0.8113. Right is pure, H = 0.

Gain = 1 − (4/6)(0.8113) − (2/6)(0) = 1 − 0.5409 = 0.459.

Gini root = 1 − (0.25 + 0.25) = 0.5. Left = 1 − (9/16 + 1/16) = 0.375. Weighted = (4/6)(0.375) = 0.25. Reduction = 0.25.

### 10 · Regularization penalties

Unit 10

β₀ = 4 (intercept), β = \[3, −1, 0.5\], λ = 0.1. Compute the ridge and lasso penalty terms, and each penalty's gradient with respect to β₁.

L2 penalty

L1 penalty

L2 grad wrt β₁

L1 grad wrt β₁

Solution

The intercept isn't penalized. L2: 0.1(9 + 1 + 0.25) = 1.025. L1: 0.1(3 + 1 + 0.5) = 0.45.

L2 gradient: 2λβ₁ = 0.6 (it would be 0.1 for β₃ = 0.5: shrinking slows as weights get small). L1 gradient: λ·sign(β₁) = 0.1, the same for every non-zero weight.

### 11 · Cross-validation bookkeeping

Unit 5

You have 250 labelled examples. With 10-fold CV, how many models are trained and how many examples does each train on? With leave-one-out?

10-fold: models

10-fold: train size

LOOCV: models

LOOCV: train size

Solution

10 folds of 25: each model trains on 225 and tests on 25. LOOCV: K = n = 250 models, each trained on 249 and tested on 1. Report the average metric across folds.

### 12 · kNN by hand

Unit 2

Query point (1, 1). Training points: A (0, 0) red, B (2, 2) blue, C (1, 3) blue, D (0, 1) red, E (3, 1) blue. Use Euclidean distance. Enter 1 for red, 2 for blue.

distance to A

k = 1 → class

k = 3 → class

k = 5 → class

Solution

Distances: A √2 = 1.414, B √2 = 1.414, C 2, D 1, E 2.

k = 1: D → red. k = 3: D, A, B → red, red, blue → red. k = 5: all → 2 red, 3 blue → blue. The prediction depends on the hyperparameter k.

### 13 · One gradient-boosting step

Unit 14

Leaf predictions \[9, 7, 6\], true values \[10, 8, 1\] (the XGBoost slide). A new tree predicts the MSE gradients exactly. With learning rate 0.5, what are the updated predictions, and the MSE before and after?

new pred 1

new pred 2

new pred 3

MSE before

MSE after

Solution

Gradients g = predicted − true = \[−1, −1, 5\]. Step against them: ŷ − 0.5·g = \[9.5, 7.5, 3.5\].

MSE before = (1 + 1 + 25)/3 = 9. After = (0.25 + 0.25 + 6.25)/3 = 2.25. Each new tree chips away at the remaining error.

All units

## Concept quiz

0 / 20 answered

Pick an answer to see whether it's right and why.

1\. Which of these is a hyperparameter?

η is a setting you choose; the others are learned from data.

2\. Training loss keeps falling but validation loss has started rising. What's happening?

The model is specializing to the training data. Patience-based early stopping would halt here.

3\. 90% of test examples are negative and your classifier has 88% accuracy. Compared with the majority baseline it is…

Always predicting "negative" scores 90%.

4\. Which penalty can set coefficients exactly to zero?

L1's gradient λ·sign(β) doesn't fade as β → 0, so small weights get pushed all the way to zero.

5\. Increasing ridge's λ generally…

Regularizing trades lower variance for higher bias.

6\. Why add a column of ones to X before solving OLS?

β₀ multiplies a constant 1 in every row.

7\. A hard-margin SVM has no solution when…

Hard margin requires every point outside the margin. With overlap you need slack (soft margin).

8\. In scikit-learn's SVC, increasing C tends to…

sklearn uses the MML meaning: C multiplies the hinge loss. (In ISL, C is a budget and works the opposite way.)

9\. Lowering γ in an RBF kernel makes the decision boundary…

Similarity exp(−γ‖x−x'‖²) decays more slowly, so distant points still count.

10\. Which training points determine an SVM's boundary?

Points beyond the margin have αᵢ = 0; moving them changes nothing.

11\. Entropy of a node whose labels are {1, 1, 1, 1}?

Pure node: no uncertainty. −1·log₂1 = 0.

12\. What makes a random forest different from plain bagging?

This decorrelates the trees and acts as a regularizer.

13\. Boosting differs from bagging because boosting…

Bagging = independent; boosting = sequential.

14\. Best encoding for "nationality"?

Nominal data has no order; integers would invent one.

15\. Temperature in °C and calendar dates are what kind of data?

Numeric with meaningful differences but no true zero.

16\. Mini-batch SGD with batch size 32 is a compromise between…

Stability vs speed.

17\. Logistic regression's cross-entropy loss is convex. What does that buy you?

No local-minimum traps (with a sensible learning rate).

18\. In the SVM dual, what does the constraint Σ yᵢαᵢ = 0 express?

You need points from both classes to define the boundary. (0 ≤ αᵢ ≤ C is the separate cap.)

19\. Precision answers which question?

Precision = TP/(TP+FP): of what you predicted positive, how much was right. The first option is recall.

20\. The chimera survey had some completion times of 0 and some over 5 hours. Best action?

These are measurement errors, not real behaviour. Contrast the sentiment-review case, where "outliers" should be kept.

Copy by hand

## Formula sheet

Your cheat sheet must be your own handwriting (one page, double-sided). Use this as the checklist of what to write down.

#### Linear regression

\\\[ y' = \\beta_0 + \\textstyle\\sum_j \\beta_jx_j \\\]\\\[ \\text{MSE} = \\tfrac1n\\textstyle\\sum(y_i - y'\_i)^2 \\\]\\\[ \\boldsymbol\\beta = (X^\\top X)^{-1}X^\\top\\mathbf y \\\]

X n×p · XᵀX p×p · β p×1

#### Gradient descent

\\\[ \\nabla\_\\beta\\text{MSE} = -\\tfrac2n X^\\top(\\mathbf y - X\\boldsymbol\\beta) \\\]\\\[ \\tfrac{\\partial}{\\partial\\beta_1} = -\\tfrac2n\\textstyle\\sum x_i(y_i - \\beta_0 - \\beta_1x_i) \\\]\\\[ \\boldsymbol\\beta \\leftarrow \\boldsymbol\\beta - \\eta\\nabla \\\]

#### Logistic regression

\\\[ \\sigma(z) = \\frac{1}{1+e^{-z}},\\ \\ \\sigma' = \\sigma(1-\\sigma) \\\]\\\[ L = -\[y\\ln\\hat y + (1-y)\\ln(1-\\hat y)\] \\\]\\\[ \\tfrac{\\partial L}{\\partial w_j} = (\\hat y - y)x_j,\\ \\ \\tfrac{\\partial L}{\\partial b} = \\hat y - y \\\]

#### Softmax

\\\[ \\text{softmax}(z_i) = \\frac{e^{z_i}}{\\sum_j e^{z_j}} \\\]

one-hot: class 2 of 5 → \[0 0 1 0 0\]

#### Metrics

\\\[ P = \\tfrac{TP}{TP+FP},\\ \\ R = \\tfrac{TP}{TP+FN} \\\]\\\[ F_1 = \\tfrac{2PR}{P+R},\\ \\ \\text{Acc} = \\tfrac{TP+TN}{\\text{all}} \\\]

macro = mean over classes; compare to majority baseline

#### Regularization

\\\[ \\text{RSS} + \\lambda\\textstyle\\sum\\beta_j^2 \\ \\ (\\text{L2, ridge}) \\\]\\\[ \\text{RSS} + \\lambda\\textstyle\\sum|\\beta_j| \\ \\ (\\text{L1, lasso}) \\\]

grad: 2λβ vs λ·sign(β) · λ→∞ ⇒ β→0

#### SVM

\\\[ \\min\\ \\tfrac12\\|w\\|^2 + C\\textstyle\\sum\\xi_i \\\]\\\[ \\xi_i = \\max(0,\\,1 - y_i(w\\cdot x_i + b)) \\\]\\\[ \\tfrac{\\partial\\ell}{\\partial w} = \\begin{cases} w & y_i(w\\cdot x_i)\\ge1\\\\ w - Cy_ix_i & \\text{else}\\end{cases} \\\]

labels ±1 · ISL C = budget · sklearn/MML C = cost

#### Kernels

\\\[ K\_{lin} = x\\cdot x' \\\]\\\[ K\_{poly} = (1 + x\\cdot x')^d \\\]\\\[ K\_{RBF} = e^{-\\gamma\\|x - x'\\|^2} \\\]\\\[ f(x) = \\beta_0 + \\textstyle\\sum\\alpha_iK(x,x_i) \\\]

#### Trees

\\\[ H = -\\textstyle\\sum p_i\\log_2p_i \\\]\\\[ IG = H(S) - \\textstyle\\sum\\frac{|S_v|}{|S|}H(S_v) \\\]\\\[ G = 1 - \\textstyle\\sum p_i^2 \\\]

log₂x = ln x / ln 2 · RF uses √p features

#### Pruning & bagging

\\\[ \\textstyle\\sum_m\\sum\_{x_i\\in R_m}(y_i - \\hat y\_{R_m})^2 + \\alpha|T| \\\]\\\[ \\hat f\_{avg}(x) = \\tfrac1B\\textstyle\\sum_b \\hat f^{\\,b}(x) \\\]

#### Preprocessing

\\\[ x' = (x - \\mu)/\\sigma \\\]

nominal → one-hot · ordinal → ordered ints · interval (no true 0) · ratio (true 0) · outliers: >3σ, but it depends

#### Handy numbers

σ(0)=0.5 · σ(0.2)=0.5498\
e¹=2.718 · e²=7.389\
H(½)=1 · H(¾,¼)=0.8113\
H(⅝,⅜)=0.9544 · ln2=0.6931

Built from Prof. Charles Welch's CS/SE 4AL3 slides, Lectures 1–12 (Weeks 1–5). The in-class calculations were recomputed and match the slides. Practice problems are new, in the same style.