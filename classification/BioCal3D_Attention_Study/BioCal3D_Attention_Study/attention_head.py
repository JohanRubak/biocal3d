"""Train a patch/point attention pooling block and three-class linear head."""
import argparse
from pathlib import Path

import numpy as np
from scipy.special import logsumexp, softmax
from threadpoolctl import threadpool_limits

from feature_cache import bag_path, key_for, mode_settings
from study_data import STAGES, digest, signature
from study_io import read_json, save_npz
from study_protocol import SEED, input_modes

HIDDEN = 128
BATCH_SIZE = 16
RATE = 1e-4
DECAY = 1e-5
DEFAULT_EPOCHS = 1000
DEFAULT_PATIENCE = 30
LEGACY_TRAINER_DIGEST = "99e5a906baec6cc170c9b6115c6f1a25dd4ee4ecd29719b94c41b9b919c0732c"
COSINE_TRAINER_DIGEST = "71fcf2fe1a074836c00b701b3a1ae45781592517dd62c39ad9a67231ee81c4bb"
# Keep the checkpoint key stable for logging changes. Update it when the fit changes.
STRICT_BEST_TRAINER_DIGEST = "f50a89403ed45c2b59f1e2671a491cd0bdac778a1916cfa7e2bd3f8bd9199902"
PARAMETERS = ("attention_input", "attention_bias", "attention_output", "classifier", "bias")


def load_bags(run, mode, rows, settings):
    bags = []
    for row in rows:
        path = bag_path(run, mode, row)
        with np.load(path, allow_pickle=False) as cache:
            if cache["key"].item() != key_for(settings, row):
                raise ValueError(f"Feature cache does not match its scan or settings: {path}")
            x = cache["bag"]
            if x.ndim != 2 or not x.size or not np.isfinite(x).all():
                raise ValueError(f"Invalid feature bag: {path}")
            bags.append(x)
    if len({bag.shape[1] for bag in bags}) != 1:
        raise ValueError(f"Feature width changed within {mode}.")
    return bags


def padded(bags):
    x = np.zeros((len(bags), max(map(len, bags)), bags[0].shape[1]), dtype=np.float32)
    mask = np.zeros(x.shape[:2], dtype=bool)
    for i, bag in enumerate(bags):
        x[i, :len(bag)] = bag
        mask[i, :len(bag)] = True
    return x, mask


def forward(x, mask, model):
    batch, locations, width = x.shape
    hidden = np.maximum(x.reshape(-1, width) @ model["attention_input"] + model["attention_bias"], 0)
    scores = (hidden @ model["attention_output"]).reshape(batch, locations)
    alpha = softmax(np.where(mask, scores, -np.inf), axis=1)
    pooled = (alpha[:, None, :] @ x)[:, 0, :]
    return pooled @ model["classifier"] + model["bias"], alpha, hidden, pooled


def losses(logits, labels):
    return logsumexp(logits, axis=1) - logits[np.arange(len(labels)), labels]


def gradients(x, mask, labels, weights, model):
    logits, alpha, hidden, pooled = forward(x, mask, model)
    delta = softmax(logits, axis=1)
    delta[np.arange(len(labels)), labels] -= 1
    delta *= (weights / weights.sum())[:, None]
    dp = delta @ model["classifier"].T
    dscore = alpha * ((x @ dp[..., None])[..., 0] - (pooled * dp).sum(axis=1, keepdims=True))
    dhidden = (dscore.reshape(-1, 1) * model["attention_output"]) * (hidden > 0)
    return dict(attention_input=x.reshape(-1, x.shape[-1]).T @ dhidden,
                attention_bias=dhidden.sum(axis=0), attention_output=hidden.T @ dscore.ravel(),
                classifier=pooled.T @ delta, bias=delta.sum(axis=0))


def score_bags(bags, model):
    with threadpool_limits(limits=4, user_api="blas"):
        logits = []
        for start in range(0, len(bags), BATCH_SIZE):
            x, mask = padded(bags[start:start + BATCH_SIZE])
            logits.append(forward(x, mask, model)[0])
    return np.concatenate(logits)


def fit(train, y_train, validation, y_val, epochs, mode, patience=DEFAULT_PATIENCE):
    labels = len(STAGES)
    if set(y_train) != set(range(labels)) or set(y_val) != set(range(labels)):
        raise ValueError("All cleaning stages are required in training and validation.")
    init, shuffle = [np.random.default_rng(s) for s in np.random.SeedSequence(SEED).spawn(2)]
    width = train[0].shape[1]
    model = dict(attention_input=init.normal(0, np.sqrt(2 / width), (width, HIDDEN)).astype(np.float32),
                 attention_bias=np.zeros(HIDDEN, dtype=np.float32),
                 attention_output=np.zeros(HIDDEN, dtype=np.float32),
                 classifier=init.normal(0, .01 / np.sqrt(width), (width, labels)).astype(np.float32),
                 bias=np.zeros(labels, dtype=np.float32))
    weight = (len(y_train) / (labels * np.bincount(y_train))).astype(np.float32)
    w_train, w_val = weight[y_train], weight[y_val]
    first = {name: np.zeros_like(value) for name, value in model.items()}
    second = {name: np.zeros_like(value) for name, value in model.items()}
    step, history, best, best_loss, best_epoch = 0, [], None, np.inf, 0
    with threadpool_limits(limits=4, user_api="blas"):
        for epoch in range(1, epochs + 1):
            order = shuffle.permutation(len(train))
            for start in range(0, len(train), BATCH_SIZE):
                ids = order[start:start + BATCH_SIZE]
                x, mask = padded([train[i] for i in ids])
                grad = gradients(x, mask, y_train[ids], w_train[ids], model)
                step += 1
                for name, value in model.items():
                    first[name] = .9 * first[name] + .1 * grad[name]
                    second[name] = .999 * second[name] + .001 * grad[name] ** 2
                    if "bias" not in name:
                        value *= 1 - RATE * DECAY
                    value -= RATE * (first[name] / (1 - .9 ** step)) / (
                        np.sqrt(second[name] / (1 - .999 ** step)) + 1e-8)
            train_loss = float(np.average(losses(score_bags(train, model), y_train), weights=w_train))
            val_loss = float(np.average(losses(score_bags(validation, model), y_val), weights=w_val))
            if not np.isfinite([train_loss, val_loss]).all():
                raise RuntimeError(f"Non-finite loss in {mode}, epoch {epoch}.")
            history.append((train_loss, val_loss))
            if val_loss < best_loss:
                best_loss, best_epoch = val_loss, epoch
                best = {name: value.copy() for name, value in model.items()}
            since_best = epoch - best_epoch
            early_stop = since_best >= patience and epoch < epochs
            if epoch == 1 or epoch % 5 == 0 or epoch == epochs or early_stop:
                print(f"{mode} epoch {epoch}/{epochs}: train {train_loss:.4f} · val {val_loss:.8f} · "
                      f"best {best_loss:.8f} at epoch {best_epoch} · "
                      f"epochs since best {since_best}/{patience}", flush=True)
            if early_stop:
                print(f"{mode}: early stop at epoch {epoch} after {patience} epochs without "
                      "a new best validation loss.", flush=True)
                break
    return best, np.asarray(history), best_epoch


def model_path(run, mode):
    return run / "models" / f"{mode}.npz"


def fit_key(run, settings, plan):
    epochs, patience, policy = plan
    if patience is None:
        schedule, trainer = epochs, LEGACY_TRAINER_DIGEST
    elif policy == "constant":
        schedule, trainer = plan, STRICT_BEST_TRAINER_DIGEST
    else:
        schedule, trainer = plan, COSINE_TRAINER_DIGEST
    return signature([digest(run / "cohort.json"), settings, schedule,
                      trainer, SEED, HIDDEN, BATCH_SIZE, RATE, DECAY])


def train_heads(run, epochs, harmonized=False, *, patience=DEFAULT_PATIENCE, retrain_heads=False):
    if epochs < 1:
        raise ValueError("At least one epoch is required.")
    if patience < 1:
        raise ValueError("--patience must be at least 1.")
    plan = (epochs, patience, "constant")
    rows = read_json(run / "cohort.json")["rows"]
    train = [row for row in rows if row["role"] == "train"]
    validation = [row for row in rows if row["role"] == "validation"]
    y_train, y_val = (np.array([STAGES.index(row["stage"]) for row in part]) for part in (train, validation))
    for mode in input_modes(harmonized):
        settings = mode_settings(run, mode)
        key = fit_key(run, settings, plan)
        path = model_path(run, mode)
        if path.is_file():
            with np.load(path, allow_pickle=False) as saved:
                matches = saved["key"].item() == key
            if matches:
                print(f"Reusing the validation-selected {mode} model.", flush=True)
                continue
            if not retrain_heads:
                raise ValueError(f"{mode} was fitted with different settings. Pass --retrain-heads "
                                 "to refit from the cached features, or choose a fresh --run folder.")
        best, history, epoch = fit(load_bags(run, mode, train, settings), y_train,
                                   load_bags(run, mode, validation, settings), y_val, epochs, mode, patience)
        save_npz(path, **best, history=history, best_epoch=np.array(epoch), key=np.array(key),
                 max_epochs=np.array(epochs), patience=np.array(patience), learning_rate_schedule=np.array(plan[2]))
        print(f"{mode}: selected epoch {epoch} of {len(history)} completed epochs by validation loss.", flush=True)


def read_model(run, mode, settings):
    with np.load(model_path(run, mode), allow_pickle=False) as saved:
        history, epoch = saved["history"], int(saved["best_epoch"].item())
        if "max_epochs" in saved:
            policy = (saved["learning_rate_schedule"].item() if "learning_rate_schedule" in saved
                      else float(saved["min_delta"].item()))
            plan = (int(saved["max_epochs"].item()), int(saved["patience"].item()), policy)
        else:
            plan = (len(history), None, None)
        if saved["key"].item() != fit_key(run, settings, plan) or not 1 <= epoch <= len(history) <= plan[0]:
            raise ValueError(f"Model does not match the saved split or feature configuration: {mode}.")
        model = {name: saved[name] for name in PARAMETERS}
    return model, history, epoch, plan


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--patience", type=int, default=DEFAULT_PATIENCE)
    parser.add_argument("--retrain-heads", action="store_true")
    parser.add_argument("--with-harmonized", action="store_true")
    args = parser.parse_args()
    try:
        train_heads(args.run, args.epochs, args.with_harmonized, patience=args.patience,
                    retrain_heads=args.retrain_heads)
    except (OSError, ValueError, KeyError, RuntimeError) as exc:
        parser.exit(2, f"Error: {exc}\n")
