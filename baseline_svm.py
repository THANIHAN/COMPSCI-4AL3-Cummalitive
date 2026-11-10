"""Majority-class + RBF SVM baselines for Philharmonia instrument classification.

Usage:
    pip install librosa scikit-learn joblib numpy
    python baseline_svm.py /path/to/philharmonia_root

The root is searched recursively for .mp3 files. Label = file name up to the
first underscore (e.g. flute_A4_025_forte_normal.mp3 -> flute). Byte-identical
duplicates (the english-horn copies of cor-anglais) are dropped by hash.
Features are cached to features.npz, so reruns skip extraction.
"""
import hashlib
import sys
import time
from pathlib import Path

import numpy as np
import librosa
from joblib import Parallel, delayed
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             classification_report, confusion_matrix, f1_score)
from sklearn.model_selection import StratifiedGroupKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

SR = 16000
CLIP_SEC = 2.0
N_MFCC = 20
SEED = 42
CACHE = Path("features.npz")


def extract(path):
    try:
        y, _ = librosa.load(path, sr=SR, mono=True)
    except Exception as e:
        print(f"skip {path}: {e}", file=sys.stderr)
        return None
    y, _ = librosa.effects.trim(y, top_db=40)
    n = int(SR * CLIP_SEC)
    y = np.pad(y, (0, max(0, n - len(y))))[:n]

    mfcc = librosa.feature.mfcc(y=y, sr=SR, n_mfcc=N_MFCC)
    spec = np.vstack([
        librosa.feature.spectral_centroid(y=y, sr=SR),
        librosa.feature.spectral_rolloff(y=y, sr=SR),
        librosa.feature.spectral_flatness(y=y),
        librosa.feature.zero_crossing_rate(y),
    ])
    rms = librosa.feature.rms(y=y)[0]
    attack = np.argmax(rms) * 512 / SR  # default hop length = 512

    return np.concatenate([mfcc.mean(1), mfcc.std(1),
                           spec.mean(1), spec.std(1), [attack]]).astype(np.float32)


def load_features(root):
    if CACHE.exists():
        d = np.load(CACHE, allow_pickle=True)
        return d["X"], d["y"], d["files"]

    files, seen = [], set()
    for p in sorted(Path(root).rglob("*.mp3")):
        h = hashlib.md5(p.read_bytes()).hexdigest()
        if h in seen:
            continue
        seen.add(h)
        files.append(p)
    print(f"{len(files)} unique clips, extracting features...")

    t = time.time()
    feats = Parallel(n_jobs=-1, verbose=5)(delayed(extract)(str(p)) for p in files)
    keep = [i for i, f in enumerate(feats) if f is not None]
    X = np.stack([feats[i] for i in keep])
    files = np.array([files[i].name for i in keep])
    y = np.array([f.split("_")[0] for f in files])
    print(f"extracted in {time.time() - t:.0f}s, X shape {X.shape}")
    np.savez(CACHE, X=X, y=y, files=files)
    return X, y, files


def report(name, y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    bal = balanced_accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="macro")
    print(f"{name:<16} acc {acc*100:5.1f}  bal_acc {bal*100:5.1f}  macro_f1 {f1*100:5.1f}")


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    X, y, files = load_features(args[0] if args else ".")
    print("class counts:", dict(zip(*np.unique(y, return_counts=True))))

    idx = np.arange(len(y))
    if "--group" in sys.argv:
        # All clips of the same instrument+pitch go to one split, so the model
        # never sees the same note (at another duration/dynamic) in train and test.
        groups = np.array(["_".join(f.split("_")[:2]) for f in files])
        folds = [f for _, f in StratifiedGroupKFold(n_splits=7, shuffle=True,
                                                     random_state=SEED).split(idx, y, groups)]
        te, va = folds[0], folds[1]
        tr = np.concatenate(folds[2:])
        assert not set(groups[tr]) & set(groups[te])
        print("GROUPED split by instrument+pitch")
    else:
        tr, tmp = train_test_split(idx, test_size=0.30, stratify=y, random_state=SEED)
        va, te = train_test_split(tmp, test_size=0.50, stratify=y[tmp], random_state=SEED)
    print(f"train {len(tr)}  val {len(va)}  test {len(te)}")

    scaler = StandardScaler().fit(X[tr])
    Xtr, Xva, Xte = scaler.transform(X[tr]), scaler.transform(X[va]), scaler.transform(X[te])

    # Grid search on the fixed validation split, selecting by macro-F1
    d = X.shape[1]
    best = (-1, None)
    for C in [0.1, 1, 10, 100]:
        for g in [1e-3, 1e-2, 1e-1, 1 / d]:
            clf = SVC(kernel="rbf", C=C, gamma=g, class_weight="balanced").fit(Xtr, y[tr])
            f1 = f1_score(y[va], clf.predict(Xva), average="macro")
            print(f"C={C:<6} gamma={g:.4f}  val macro-F1 {f1*100:.1f}")
            if f1 > best[0]:
                best = (f1, (C, g))
    C, g = best[1]
    print(f"\nbest: C={C}, gamma={g:.4f} (val macro-F1 {best[0]*100:.1f})")

    # Final model: refit on train only (val stays clean for later CNN comparison)
    svm = SVC(kernel="rbf", C=C, gamma=g, class_weight="balanced").fit(Xtr, y[tr])
    maj = DummyClassifier(strategy="most_frequent").fit(Xtr, y[tr])

    print("\n=== TEST ===")
    report("majority", y[te], maj.predict(Xte))
    pred = svm.predict(Xte)
    report("svm", y[te], pred)
    print("\n" + classification_report(y[te], pred, digits=3))

    labels = np.unique(y)
    cm = confusion_matrix(y[te], pred, labels=labels)
    np.savetxt("confusion_matrix.csv", cm, fmt="%d", delimiter=",",
               header=",".join(labels), comments="")
    off = cm.copy()
    np.fill_diagonal(off, 0)
    print("top confusions (true -> predicted):")
    for k in np.argsort(off, axis=None)[::-1][:8]:
        i, j = divmod(k, len(labels))
        if off[i, j] == 0:
            break
        print(f"  {labels[i]} -> {labels[j]}: {off[i, j]}")

    if "--ablate" not in sys.argv:
        return

    # ---- Feature ablation, scored on VALIDATION so we can select features ----
    # Layout: 0-19 MFCC means, 20-39 MFCC stds, 40-43 spectral means
    # (centroid, rolloff, flatness, zcr), 44-47 spectral stds, 48 attack.
    G = {
        "mfcc":     list(range(0, 2 * N_MFCC)),
        "centroid": [40, 44],
        "rolloff":  [41, 45],
        "flatness": [42, 46],
        "zcr":      [43, 47],
        "attack":   [48],
    }

    def val_f1(cols, test=False):
        cols = sorted(cols)
        # gamma scaled by dimension so the kernel width stays comparable
        m = SVC(kernel="rbf", C=C, gamma=g * d / len(cols),
                class_weight="balanced").fit(Xtr[:, cols], y[tr])
        Xe, ye = (Xte, y[te]) if test else (Xva, y[va])
        return f1_score(ye, m.predict(Xe[:, cols]), average="macro") * 100

    all_cols = sum(G.values(), [])
    base = val_f1(all_cols)
    print(f"\n=== ABLATION (val macro-F1, all features = {base:.1f}) ===")

    print("-- leave one group out --")
    for name in G:
        cols = [c for n, cs in G.items() if n != name for c in cs]
        f = val_f1(cols)
        print(f"  no {name:<9} {f:5.1f}  ({f - base:+.1f})")

    print("-- only one group --")
    for name, cs in G.items():
        print(f"  only {name:<7} {val_f1(cs):5.1f}")

    print("-- number of MFCCs (others kept) --")
    rest = [c for n, cs in G.items() if n != "mfcc" for c in cs]
    for k in [5, 10, 13, 20]:
        cols = list(range(k)) + list(range(N_MFCC, N_MFCC + k)) + rest
        print(f"  {k:>2} MFCCs  {val_f1(cols):5.1f}")

    print("-- MFCC means only vs stds only --")
    print(f"  means only {val_f1(list(range(N_MFCC)) + rest):5.1f}")
    print(f"  stds only  {val_f1(list(range(N_MFCC, 2 * N_MFCC)) + rest):5.1f}")

    # Greedy backward elimination over groups: drop a group while val F1 doesn't fall
    print("-- backward elimination --")
    keep, cur = list(G), base
    while len(keep) > 1:
        trials = {n: val_f1([c for k in keep if k != n for c in G[k]]) for n in keep}
        n, f = max(trials.items(), key=lambda kv: kv[1])
        if f < cur:
            break
        keep.remove(n)
        cur = f
        print(f"  drop {n:<9} -> {f:5.1f}")
    print(f"  selected: {keep} (val {cur:.1f})")

    sel = [c for k in keep for c in G[k]]
    print(f"\n=== TEST, selected features ===\n  macro-F1 {val_f1(sel, test=True):.1f} "
          f"(all features: {val_f1(all_cols, test=True):.1f})")


if __name__ == "__main__":
    main()