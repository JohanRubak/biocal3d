"""Train a small MIL pooling head on frozen BioCal3D features."""

import numpy as np
from scipy.special import expit, softmax
from threadpoolctl import threadpool_limits

from study_data import STAGES


def normalize_local(features):
    """Unit RMS per location, with a floor for zero vectors."""
    rms = np.sqrt(np.einsum("ij,ij->i", features, features) / features.shape[1])
    return np.ascontiguousarray(features / np.maximum(rms[:, None], 1e-6))


def forward(bag, model):
    x, prior = bag
    hidden = None
    if "attention_input" in model:
        hidden = np.maximum(x @ model["attention_input"] + model["attention_bias"], 0)
        # A common output bias cancels in softmax. The offset accounts for
        # partial DINOv3 patch coverage and is constant for Concerto points.
        attention = softmax(hidden @ model["attention_output"] + np.log(prior))
    else:
        attention = prior / prior.sum()
    pooled = attention @ x
    logits = pooled @ model["classifier"] + model["bias"]
    return attention, pooled, logits, hidden


def loss_gradient(bag, target, model, stage_weight, proxy_weight):
    attention, pooled, logits, hidden = forward(bag, model)
    probabilities = softmax(logits[:-1])
    loss = stage_weight * (np.logaddexp.reduce(logits[:-1]) - logits[target])
    delta = np.empty_like(logits)
    delta[:-1] = probabilities
    delta[target] -= 1
    delta[:-1] *= stage_weight
    delta[-1] = 0
    if proxy_weight:
        positive = target == STAGES.index("baseline")
        loss += proxy_weight * (np.logaddexp(0, logits[-1]) - positive * logits[-1])
        delta[-1] = proxy_weight * (expit(logits[-1]) - positive)
    gradient = {"classifier": np.outer(pooled, delta), "bias": delta}
    if hidden is not None:
        x, _ = bag
        dp = model["classifier"] @ delta
        dscore = attention * (x @ dp - pooled @ dp)
        dhidden = np.outer(dscore, model["attention_output"]) * (hidden > 0)
        gradient["attention_output"] = hidden.T @ dscore
        gradient["attention_input"] = x.T @ dhidden
        gradient["attention_bias"] = dhidden.sum(axis=0)
    return float(loss), gradient


def fit(bags, labels, seed, epochs, rate, batch_size, proxy_strength, hidden_dim):
    """Fit the stage classifier and endpoint proxy jointly with minibatch Adam."""
    init_seed, shuffle_seed = np.random.SeedSequence(seed).spawn(2)
    init = np.random.default_rng(init_seed)
    shuffle = np.random.default_rng(shuffle_seed)
    size = bags[0][0].shape[1]
    model = {
        "classifier": init.normal(0, .01 / np.sqrt(size), (size, len(STAGES) + 1)).astype(np.float32),
        "bias": np.zeros(len(STAGES) + 1, dtype=np.float32),
    }
    if hidden_dim:
        model.update(
            attention_input=init.normal(0, np.sqrt(2 / size), (size, hidden_dim)).astype(np.float32),
            attention_bias=np.zeros(hidden_dim, dtype=np.float32),
            attention_output=np.zeros(hidden_dim, dtype=np.float32),
        )
    if set(labels) != set(range(len(STAGES))):
        raise ValueError("Training scans must contain all three stages.")
    stage_weights = len(labels) / (len(STAGES) * np.bincount(labels))
    proxy_weights = np.zeros(len(STAGES))
    for stage in ("baseline", "clean"):
        i = STAGES.index(stage)
        proxy_weights[i] = proxy_strength * len(labels) / (2 * np.sum(labels == i))
    m = {key: np.zeros_like(value) for key, value in model.items()}
    v = {key: np.zeros_like(value) for key, value in model.items()}
    step, history = 0, []
    with threadpool_limits(limits=1, user_api="blas"):
        for _ in range(epochs):
            permutation = shuffle.permutation(len(bags))
            epoch_loss = 0.
            for start in range(0, len(bags), batch_size):
                batch = permutation[start:start + batch_size]
                gradients = {key: np.zeros_like(value) for key, value in model.items()}
                for i in batch:
                    target = labels[i]
                    loss, grad = loss_gradient(bags[i], target, model, stage_weights[target],
                                               proxy_weights[target])
                    epoch_loss += loss
                    for key, value in grad.items():
                        gradients[key] += value
                step += 1
                for key, value in model.items():
                    grad = gradients[key] / len(batch)
                    if not key.endswith("bias"):
                        grad += .001 * value
                    m[key] = .9 * m[key] + .1 * grad
                    v[key] = .999 * v[key] + .001 * grad**2
                    value -= rate * (m[key] / (1 - .9**step)) / (
                        np.sqrt(v[key] / (1 - .999**step)) + 1e-8)
            history.append(epoch_loss / len(bags))
    if not np.isfinite(history).all() or any(not np.isfinite(v).all() for v in model.values()):
        raise ValueError("Attention fitting diverged. Try a smaller --learning-rate.")
    return model, history


def clean_threshold(model, bags, labels, quantile):
    """Estimate a weighted local-logit quantile of clean training scans."""
    with threadpool_limits(limits=1, user_api="blas"):
        clean = [bag for bag, label in zip(bags, labels) if label == STAGES.index("clean")]
        local = np.concatenate([x @ model["classifier"][:, -1] + model["bias"][-1]
                                for x, _ in clean])
    weights = np.concatenate([prior / prior.sum(dtype=np.float64) for _, prior in clean])
    order = np.argsort(local, kind="stable")
    mass = np.cumsum(weights[order], dtype=np.float64)
    index = np.searchsorted(mass, quantile * mass[-1])
    return float(local[order[index]])
