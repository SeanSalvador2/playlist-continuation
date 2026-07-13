"""Fit-once / assign-many taste *flavors* for change-point work.

WHY THIS EXISTS (and why we do NOT reuse ``TasteEngine.flavor_clusters``)
------------------------------------------------------------------------
The taste engine distils a user's tracks into named "flavor" clusters, but it
**refits k-means on every call** (``KMeans(...).fit`` inside
:meth:`playlistcont.models.taste_engine.TasteEngine.flavor_clusters`).  For a
one-shot recommendation that is fine.  For change-point detection it is fatal:
if we clustered each analysis window separately, cluster #2 in January need not
be the same taste as cluster #2 in June — the identities *flap*, and a "flavor
share" time series built from them would be pure label noise.  We must not
modify the engine (it is a frozen deliverable), so this module implements the
discipline the detector needs:

    fit ONCE on the union of the listener's track features, then ASSIGN every
    window's plays to those frozen centroids.

Because the centroids never move, "share of plays in flavor *c*" is comparable
across windows, which is the entire point.  We reuse the engine's
:func:`~playlistcont.models.taste_engine.name_flavor` for the English labels so
the naming convention stays identical to the rest of the project.

CONVENTIONS
-----------
* We cluster on the **scalar taste axes only** (``SCALAR_AXES``; tempo, energy,
  valence, acousticness, lyrical-depth) — the continuous "mood" space.  The
  genre block is a near-one-hot and would dominate a Euclidean k-means; it is
  already captured by the genre-share columns of the window matrix.
* :func:`fit_flavors` uses ``sklearn.cluster.KMeans`` with an **explicit**
  ``random_state`` and ``n_init`` so a given ``(features, k, seed)`` triple is
  perfectly reproducible cross-process (the determinism contract the rest of
  the repo holds itself to).
* Cluster **names** come from :func:`name_flavor` applied to the cluster's mean
  *full* axis vector when the caller supplies ``full_features`` (so the name can
  mention a genre), otherwise to the scalar centroid padded to full length.
  Names are for humans; the *column order is the integer cluster id*, which is
  what stays stable across windows.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from ..data.schema import N_AXES, SCALAR_AXES
from ..models.taste_engine import name_flavor

N_SCALAR = len(SCALAR_AXES)


@dataclass
class FlavorModel:
    """A frozen k-means over scalar taste axes, fit once, assigned many times.

    * ``centroids`` — ``(k, N_SCALAR)`` cluster centres in raw scalar-axis space.
    * ``names`` — English label per cluster id (``name_flavor`` convention).
    * ``k`` / ``seed`` — the fit parameters, retained for reproducibility.

    The integer cluster id (a column position) is the stable identity; the name
    is a human-readable decoration of it.
    """

    kmeans: object
    centroids: np.ndarray
    names: List[str]
    k: int
    seed: int

    def assign(self, features: np.ndarray) -> np.ndarray:
        """Nearest-centroid label for each row of ``features`` (``(n, N_SCALAR)``)."""
        return assign(self, features)


def _as_scalar_matrix(features: np.ndarray) -> np.ndarray:
    """Coerce an ``(n, N_SCALAR)`` or ``(n, N_AXES)`` matrix to the scalar block."""
    X = np.asarray(features, dtype=float)
    if X.ndim != 2:
        raise ValueError(f"expected a 2-D feature matrix, got shape {X.shape}")
    if X.shape[1] == N_SCALAR:
        return X
    if X.shape[1] >= N_AXES:
        return X[:, :N_SCALAR]
    raise ValueError(
        f"feature matrix has {X.shape[1]} columns; expected {N_SCALAR} (scalar) "
        f"or >= {N_AXES} (full axis vector)")


def fit_flavors(
    features_matrix: np.ndarray,
    k: int = 4,
    seed: int = 0,
    full_features: Optional[np.ndarray] = None,
    n_init: int = 10,
) -> FlavorModel:
    """Fit ``k`` flavor centroids ONCE on the union of a listener's track features.

    ``features_matrix`` is the listener's per-track scalar-axis matrix
    (``(n_tracks, N_SCALAR)``; a full ``(n_tracks, N_AXES)`` matrix is accepted and
    its scalar block used).  ``k`` is clamped to the number of distinct tracks.
    ``full_features`` (``(n_tracks, N_AXES)``), when given, is used only to *name*
    the clusters via :func:`name_flavor` (so a name can mention the dominant
    genre); it never affects the clustering.

    Determinism: ``random_state=seed`` and an explicit ``n_init`` (default 10) —
    the same inputs always yield the same centroids and the same cluster ids.
    """
    from sklearn.cluster import KMeans

    X = _as_scalar_matrix(features_matrix)
    n = X.shape[0]
    k_eff = int(max(1, min(k, n)))
    km = KMeans(n_clusters=k_eff, n_init=n_init, random_state=seed)
    labels = km.fit_predict(X)
    centroids = np.asarray(km.cluster_centers_, dtype=float)

    # Name each cluster from the mean FULL axis vector of its members when the
    # caller handed us the full features; otherwise pad the scalar centroid.
    names: List[str] = []
    for c in range(k_eff):
        if full_features is not None:
            members = np.asarray(full_features, dtype=float)[labels == c]
            centroid_full = (members.mean(axis=0) if len(members)
                             else _pad_to_full(centroids[c]))
        else:
            centroid_full = _pad_to_full(centroids[c])
        names.append(name_flavor(centroid_full))
    return FlavorModel(kmeans=km, centroids=centroids, names=names,
                       k=k_eff, seed=seed)


def _pad_to_full(scalar_centroid: np.ndarray) -> np.ndarray:
    """Pad a scalar centroid to length ``N_AXES`` (zeros for the genre block).

    :func:`name_flavor` reads ``centroid[:5]`` for the mood words and
    ``centroid[5:]`` for the genre; an all-zero genre block simply yields no
    genre word, which is the honest thing when we clustered on mood alone.
    """
    full = np.zeros(N_AXES, dtype=float)
    full[:N_SCALAR] = scalar_centroid[:N_SCALAR]
    return full


def assign(model: FlavorModel, features: np.ndarray) -> np.ndarray:
    """Assign rows of ``features`` to ``model``'s frozen clusters (integer ids).

    Uses the fitted estimator's ``predict`` — i.e. nearest centroid in the same
    scalar-axis space the model was fit in.  An empty input returns an empty
    integer array.
    """
    X = _as_scalar_matrix(features)
    if X.shape[0] == 0:
        return np.empty(0, dtype=int)
    return model.kmeans.predict(X).astype(int)
