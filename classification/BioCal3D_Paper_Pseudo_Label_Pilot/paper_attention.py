"""Frozen-feature attention pooling with binary scan-level supervision."""

import numpy as np
from scipy.special import expit, softmax
from threadpoolctl import threadpool_limits


def normalize(features):
    x = np.asarray(features, dtype=np.float32)
    rms = np.sqrt(np.einsum("ij,ij->i", x, x) / x.shape[1])
    return np.ascontiguousarray(x / np.maximum(rms[:, None], 1e-6))


def forward(bag, model):
    x, prior = bag
    hidden = np.maximum(x @ model["attention_input"] + model["attention_bias"], 0)
    attention = softmax(hidden @ model["attention_output"] + np.log(prior))
    pooled = attention @ x
    logit = float(pooled @ model["classifier"] + model["bias"][0])
    return attention, pooled, logit, hidden


def loss_gradient(bag, target, model, weight):
    x, _ = bag
    attention, pooled, logit, hidden = forward(bag, model)
    delta = weight * (expit(logit) - target)
    dp = model["classifier"] * delta
    dscore = attention * (x @ dp - pooled @ dp)
    dhidden = np.outer(dscore, model["attention_output"]) * (hidden > 0)
    return float(weight * (np.logaddexp(0, logit) - target * logit)), {
        "classifier": pooled * delta,
        "bias": np.array([delta], dtype=np.float32),
        "attention_output": hidden.T @ dscore,
        "attention_input": x.T @ dhidden,
        "attention_bias": dhidden.sum(axis=0),
    }


def fit(bags, labels, seed, epochs, hidden_dim, rate=.001, batch_size=8):
    if set(labels) != {0, 1} or len(bags) != len(labels):
        raise ValueError("Training requires plaque-stage and final-clean scan labels.")
    init_seed, shuffle_seed = np.random.SeedSequence(seed).spawn(2)
    init, shuffle = np.random.default_rng(init_seed), np.random.default_rng(shuffle_seed)
    size = bags[0][0].shape[1]
    model = {
        "classifier": init.normal(0, .01 / np.sqrt(size), size).astype(np.float32),
        "bias": np.zeros(1, dtype=np.float32),
        "attention_input": init.normal(0, np.sqrt(2 / size), (size, hidden_dim)).astype(np.float32),
        "attention_bias": np.zeros(hidden_dim, dtype=np.float32),
        "attention_output": np.zeros(hidden_dim, dtype=np.float32),
    }
    weights = len(labels) / (2 * np.bincount(labels))
    first = {key: np.zeros_like(value) for key, value in model.items()}
    second = {key: np.zeros_like(value) for key, value in model.items()}
    step, history = 0, []
    with threadpool_limits(limits=1, user_api="blas"):
        for _ in range(epochs):
            order = shuffle.permutation(len(bags))
            epoch_loss = 0.
            for start in range(0, len(order), batch_size):
                batch = order[start:start + batch_size]
                gradients = {key: np.zeros_like(value) for key, value in model.items()}
                for index in batch:
                    target = labels[index]
                    loss, gradient = loss_gradient(bags[index], target, model, weights[target])
                    epoch_loss += loss
                    for key, value in gradient.items():
                        gradients[key] += value
                step += 1
                for key, value in model.items():
                    gradient = gradients[key] / len(batch)
                    if not key.endswith("bias"):
                        gradient += .001 * value
                    first[key] = .9 * first[key] + .1 * gradient
                    second[key] = .999 * second[key] + .001 * gradient**2
                    value -= rate * (first[key] / (1 - .9**step)) / (
                        np.sqrt(second[key] / (1 - .999**step)) + 1e-8)
            history.append(epoch_loss / len(bags))
    if not np.isfinite(history).all() or any(not np.isfinite(value).all() for value in model.values()):
        raise ValueError("Attention fitting produced non-finite values.")
    return model, history


def attention_to_unit_range(attention, support):
    result = np.full(attention.shape, np.nan, dtype=np.float32)
    values = attention[support]
    low, high = np.quantile(values, [.01, .99])
    if high - low <= 1e-12:
        result[support] = 0.
    else:
        result[support] = np.clip((values - low) / (high - low), 0, 1)
    return result


def refine_local(unary, support, luminance, iterations=5, strength=1.0):
    """Mean-field refinement on a four-neighbour ROI graph with optional image edges."""
    base = np.clip(np.nan_to_num(unary, nan=.5), 1e-4, 1 - 1e-4)
    unary_logit = np.log(base / (1 - base))
    probability = base.copy()
    edges = []
    for axis in (0, 1):
        a = [slice(None), slice(None)]
        b = [slice(None), slice(None)]
        a[axis], b[axis] = slice(None, -1), slice(1, None)
        a, b = tuple(a), tuple(b)
        weights = (support[a] & support[b]).astype(np.float32)
        if luminance is not None:
            weights *= np.exp(-.5 * ((luminance[a] - luminance[b]) / .12)**2)
        edges.append((a, b, weights))
    for _ in range(iterations):
        signal = np.zeros_like(probability)
        for a, b, weights in edges:
            signal[a] += weights * (2 * probability[b] - 1)
            signal[b] += weights * (2 * probability[a] - 1)
        probability = expit(unary_logit + strength * signal)
    return np.where(support, probability, np.nan).astype(np.float32)
