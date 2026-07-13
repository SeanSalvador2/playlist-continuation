"""A 2-D *taste trajectory* — PCA over the window feature time series (Phase 4).

The change-point package (Phase 3) answers *when* taste changed.  This module
answers *what the shape of the whole journey looks like*: it projects the
per-window taste vectors down to two interpretable axes so the UI can draw the
listener's path through "taste space" as a single connected line, with the
windows as time-ordered points and the detected changes as markers.

WHAT IT DOES
------------
:func:`compute_trajectory` takes a :class:`~playlistcont.dynamics.windows.WindowSeries`,
keeps the **unmasked** windows of the ``combined`` taste representation (5 mood
axes + 10 genre shares + k flavor shares), **standardises each column** to mean 0
/ unit variance (population std, so a genre-share move is weighed against a
valence move by its own variability — the same discipline the detectors use), and
runs an ordinary PCA (``sklearn.decomposition.PCA``, exact full SVD).  It exposes,
per window, the 2-D coordinates; the explained-variance ratio of each component;
and the **top loadings** of each component mapped back to human-readable column
names, so the product can caption an axis as e.g. ``"PC1 ≈ energy + metal share
vs acousticness"``.

DETERMINISM & THE SIGN CONVENTION (documented, tested)
------------------------------------------------------
PCA component signs are arbitrary — a sign flip is an equally valid solution, and
different runs/BLAS builds can return either.  To make the projected path stable
run-to-run we fix a convention: **each component is oriented so that its
largest-magnitude loading is positive**.  Concretely, for each component we find
the column with the biggest ``|loading|``; if that loading is negative we negate
the whole component (loadings *and* the corresponding window scores).  With a
fixed ``svd_solver="full"`` and this orientation, ``compute_trajectory`` is
bit-for-bit reproducible across calls.  (``random_state`` is also pinned, though
the full solver is already deterministic.)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Dict, List

import numpy as np

from ..data.schema import SCALAR_AXES
from ..history.store import feature_column
from .flavors import FlavorModel
from .windows import WindowSeries

STD_FLOOR = 1e-9

_SCALAR_LABEL: Dict[str, str] = {
    "tempo": "tempo", "energy": "energy", "valence": "valence",
    "acousticness": "acousticness", "lyrical_depth": "lyrical depth",
}
_SCALAR_COLUMNS = {feature_column(a): a for a in SCALAR_AXES}


def human_label(column: str, flavor_model: FlavorModel | None = None) -> str:
    """Human-readable name for a window-matrix column.

    * scalar-axis columns -> the mood-axis word (``lyrical_depth`` -> ``"lyrical depth"``);
    * ``genre_<g>`` -> ``"<g> share"`` (e.g. ``genre_metal`` -> ``"metal share"``);
    * ``flavor_<i>`` -> the frozen flavor's English name when a ``flavor_model`` is
      supplied, else ``"flavor <i>"``.
    """
    if column in _SCALAR_COLUMNS:
        return _SCALAR_LABEL[_SCALAR_COLUMNS[column]]
    if column.startswith("genre_"):
        return f"{column[len('genre_'):]} share"
    if column.startswith("flavor_"):
        idx = int(column[len("flavor_"):])
        if flavor_model is not None and idx < len(flavor_model.names):
            return flavor_model.names[idx]
        return f"flavor {idx}"
    return column


@dataclass
class Component:
    """One principal component: its explained variance and top human-named loadings."""

    index: int                       # 0-based component number (PC1 = 0)
    explained_variance_ratio: float
    loadings: List[dict]             # [{column, label, loading}], sorted by |loading| desc
    caption: str                     # e.g. "energy + metal share vs acousticness"


@dataclass
class Trajectory:
    """A 2-D PCA projection of a listener's window taste vectors.

    * ``points`` — one per unmasked window, ``{date, coords: [x, y, ...],
      event_count}`` in calendar order (the drawing order of the path).
    * ``components`` — :class:`Component` per retained axis (loadings + caption).
    * ``columns`` — the human labels of the input columns (for reference).
    * ``n_components`` / ``n_windows`` — retained dimensionality and point count.
    """

    points: List[dict]
    components: List[Component]
    columns: List[str]
    n_components: int
    n_windows: int
    total_explained: float = 0.0

    def to_payload(self) -> dict:
        return {
            "points": self.points,
            "components": [
                {"index": c.index,
                 "explained_variance_ratio": c.explained_variance_ratio,
                 "loadings": c.loadings, "caption": c.caption}
                for c in self.components
            ],
            "columns": self.columns,
            "n_components": self.n_components,
            "n_windows": self.n_windows,
            "total_explained": self.total_explained,
        }


def _caption(loadings: List[dict], top: int = 3) -> str:
    """'energy + metal share vs acousticness' from the top-|loading| columns."""
    picks = loadings[:top]
    pos = [d["label"] for d in picks if d["loading"] > 0]
    neg = [d["label"] for d in picks if d["loading"] < 0]
    pos_s = " + ".join(pos)
    neg_s = " + ".join(neg)
    if pos_s and neg_s:
        return f"{pos_s} vs {neg_s}"
    return pos_s or neg_s or "(flat)"


def compute_trajectory(ws: WindowSeries, n_components: int = 2) -> Trajectory:
    """Project the unmasked ``combined`` window vectors to ``n_components`` PCA axes.

    Standardises each column of the combined taste representation over the unmasked
    windows, runs exact PCA, and orients every component so its largest-magnitude
    loading is positive (see the module docstring for the determinism contract).
    Returns a :class:`Trajectory` with per-window coordinates, explained-variance
    ratios and human-named top loadings per component.
    """
    from sklearn.decomposition import PCA

    cols = ws.representation_columns("combined")
    idx = ws.valid_indices()
    raw = ws.submatrix("combined")[idx]
    dates: List[date] = [ws.starts[i] for i in idx]
    counts = [int(ws.event_counts[i]) for i in idx]

    labels = [human_label(c, ws.flavor_model) for c in cols]
    n_valid, n_feat = raw.shape
    k = int(max(1, min(n_components, n_feat, max(1, n_valid))))

    # empty / degenerate: nothing to project
    if n_valid < 2 or n_feat == 0:
        return Trajectory(points=[], components=[], columns=labels,
                          n_components=0, n_windows=n_valid)

    mu = raw.mean(axis=0)
    sd = raw.std(axis=0)
    sd = np.where(sd < STD_FLOOR, 1.0, sd)
    Z = (raw - mu) / sd

    pca = PCA(n_components=k, svd_solver="full", random_state=0)
    scores = pca.fit_transform(Z)               # (n_valid, k)
    comps = np.asarray(pca.components_, dtype=float).copy()   # (k, n_feat)
    evr = np.asarray(pca.explained_variance_ratio_, dtype=float)

    # ---- sign convention: largest-|loading| column positive -------------- #
    for c in range(k):
        lead = int(np.argmax(np.abs(comps[c])))
        if comps[c, lead] < 0:
            comps[c] *= -1.0
            scores[:, c] *= -1.0

    components: List[Component] = []
    for c in range(k):
        order = np.argsort(-np.abs(comps[c]))
        loadings = [
            {"column": cols[j], "label": labels[j], "loading": float(comps[c, j])}
            for j in order
        ]
        components.append(Component(
            index=c, explained_variance_ratio=float(evr[c]),
            loadings=loadings, caption=_caption(loadings)))

    points = [
        {"date": dates[i].isoformat(),
         "coords": [float(v) for v in scores[i, :k]],
         "event_count": counts[i]}
        for i in range(n_valid)
    ]
    return Trajectory(
        points=points, components=components, columns=labels,
        n_components=k, n_windows=n_valid,
        total_explained=float(evr[:k].sum()))
