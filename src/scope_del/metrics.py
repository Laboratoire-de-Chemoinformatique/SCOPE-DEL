import numpy as np
from numba import njit, prange
from typing import Optional, Tuple, Union, List, Dict
import pandas as pd


@njit(parallel=True, fastmath=True)
def tanimoto_int_similarity_matrix_numba(
    v_a: np.ndarray, v_b: np.ndarray
) -> np.ndarray:
    """
    Implement the Tanimoto similarity measure for integer matrices, comparing each vector in v_a against each in v_b.

    Parameters:
    - v_a (np.ndarray): Numpy matrix where each row represents a vector a.
    - v_b (np.ndarray): Numpy matrix where each row represents a vector b.

    Returns:
    - np.ndarray: Matrix of computed similarity scores, where element (i, j) is the similarity between row i of v_a and row j of v_b.
    """

    num_rows_a = v_a.shape[0]
    num_rows_b = v_b.shape[0]
    similarity_matrix = np.empty((num_rows_a, num_rows_b), dtype=np.float64)

    sum_a_squared = np.sum(np.square(v_a), axis=1)
    sum_b_squared = np.sum(np.square(v_b), axis=1)

    for i in prange(num_rows_a):
        for j in prange(num_rows_b):
            numerator = np.dot(v_a[i], v_b[j])
            denominator = sum_a_squared[i] + sum_b_squared[j] - numerator

            if denominator == 0:
                similarity = 0.0
            else:
                similarity = numerator / denominator

            # similarity_matrix[i, j] = similarity
            similarity_matrix[i, j] = max(1 - similarity, 0)

    return similarity_matrix

def top_overlap(
    s_a: pd.Series,
    s_b: pd.Series,
    n_a: Optional[int] = None,      # e.g., 5 or 10
    pct_a: Optional[float] = None,   # e.g., 0.05 or 0.10
    n_b: Optional[int] = None,
    pct_b: Optional[float] = None,
    dropna: bool = True,
) -> Tuple[int, set, set, set]:
    """
    Count overlap between the top of s_a and the top of s_b.
    Overlap is computed on index labels (IDs). Ensure s_a and s_b share the same index.
    Returns: (count, ids_top_a, ids_top_b, ids_overlap)
    """
    if dropna:
        s_a = s_a.dropna()
        s_b = s_b.dropna()

    # how many to take from each series
    def top_k(series: pd.Series, n: Optional[int], pct: Optional[float]) -> int:
        if n is not None:
            return max(1, min(n, len(series)))
        if pct is not None:
            k = int(np.ceil(len(series) * pct))
            return max(1, k)
        raise ValueError("Specify either n or pct for both A and B.")

    k_a = top_k(s_a, n_a, pct_a)
    k_b = top_k(s_b, n_b, pct_b)

    # pick top by value (ties are resolved by order; adjust as needed)
    top_a_ids = set(s_a.nlargest(k_a).index)
    top_b_ids = set(s_b.nlargest(k_b).index)

    overlap_ids = top_a_ids & top_b_ids
    return len(overlap_ids), top_a_ids, top_b_ids, overlap_ids

def positives_from_series(s: pd.Series, n: int = None, pct: float = None) -> pd.Index:
    """
    Return the index labels of the top-n or top-pct items in series s.
    """
    if (n is None) == (pct is None):
        raise ValueError("Specify exactly one of n or pct.")
    k = n if n is not None else max(1, int(np.ceil(len(s) * pct)))
    return s.nlargest(k).index


def make_labels_and_scores(s_pos: pd.Series, s_score: pd.Series,
                           n: int = None, pct: float = None):
    """
    From two aligned series (same index):
      - positives = top-n / top-pct of s_pos
      - scores    = values of s_score
    Returns y_true (0/1 ndarray) and y_score (float ndarray), both in the same order.
    """
    s_pos = s_pos.dropna()
    s_score = s_score.dropna()
    idx = s_pos.index.intersection(s_score.index)

    pos_idx = positives_from_series(s_pos.loc[idx], n=n, pct=pct)
    y_true = np.isin(idx, pos_idx).astype(int)
    y_score = s_score.loc[idx].to_numpy()
    return idx, y_true, y_score

# ----------------------------
# Enrichment Factor (ER / EF)
# ----------------------------
def enrichment_factor(y_true: np.ndarray, y_score: np.ndarray,
                      top_k: int = None, top_pct: float = None) -> float:
    """
    ER at a given cutoff on *scores*.
    ER = (positives found in top subset / size of top subset)
         / (total positives / total N)
    """
    if (top_k is None) == (top_pct is None):
        raise ValueError("Specify exactly one of top_k or top_pct.")

    N = y_true.size
    if top_k is None:
        top_k = max(1, int(np.ceil(N * top_pct)))

    order = np.argsort(-y_score)  # descending
    top_mask = np.zeros(N, dtype=bool)
    top_mask[order[:top_k]] = True

    found = y_true[top_mask].sum()
    hit_rate_top = found / top_k
    base_rate = y_true.sum() / N if y_true.sum() > 0 else 0.0
    return (hit_rate_top / base_rate) if base_rate > 0 else np.nan

# ----------------------------
# BEDROC (Truchon & Bayly, 2007)
# ----------------------------
def bedroc(y_true: np.ndarray, y_score: np.ndarray, alpha: float = 20.0) -> float:
    """
    BEDROC in [0,1], emphasizing early recognition.
    Steps:
      1) rank items by descending y_score
      2) compute RIE for observed ranking
      3) compute RIE_min (all positives at end) and RIE_max (all positives at top)
      4) BEDROC = (RIE - RIE_min) / (RIE_max - RIE_min)
    """
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score, dtype=float)

    N = y_true.size
    n = y_true.sum()
    if N == 0 or n == 0 or n == N:
        return np.nan  # undefined or degenerate

    # Sort by descending score, get positive ranks (1..N)
    order = np.argsort(-y_score)
    ranks = np.empty(N, dtype=int)
    ranks[order] = np.arange(1, N + 1)
    pos_ranks = ranks[y_true == 1]
    x = pos_ranks / N  # normalized ranks in (0,1]

    # RIE with standard normalization constant C
    C = (1.0 - np.exp(-alpha)) / (alpha)
    RIE = (np.exp(-alpha * x).mean()) / C

    # Ideal extremes (positives at very top vs very bottom)
    x_max = (np.arange(1, n + 1)) / N               # best case
    x_min = (np.arange(N - n + 1, N + 1)) / N       # worst case
    RIE_max = (np.exp(-alpha * x_max).mean()) / C
    RIE_min = (np.exp(-alpha * x_min).mean()) / C

    # Normalize to [0,1]
    if np.isclose(RIE_max, RIE_min):
        return np.nan
    return float((RIE - RIE_min) / (RIE_max - RIE_min))

def coverage_metrics(
    matrix: Union[pd.DataFrame, np.ndarray],
    threshold: float = 0.35,
    is_distance: bool = True,
    library_similarity_matrix: Optional[Union[pd.DataFrame, np.ndarray]] = None,
    calculate_intra_library_stats: bool = False,
) -> Dict[str, Union[float, int, pd.DataFrame]]:
    """
    Compute comprehensive coverage metrics for similarity or distance matrix.

    This function analyzes how well a library covers a set of targets based on
    similarity/distance thresholds, providing coverage, participation, and evenness metrics.

    Parameters
    ----------
    matrix : pd.DataFrame or np.ndarray
        Distance or similarity matrix where rows = targets, cols = library compounds.
        For distance matrices, set is_distance=True.
        For similarity matrices, set is_distance=False.
    threshold : float, optional
        Threshold value for determining hits (default: 0.35).
        For similarity: values >= threshold are considered hits.
        For distance: values <= threshold are considered hits.
    is_distance : bool, optional
        Whether the input matrix contains distances (True) or similarities (False).
        Default: True (distance matrix).
    library_similarity_matrix : pd.DataFrame or np.ndarray, optional
        Full library-vs-library similarity matrix (library compounds x library compounds).
        Required when calculate_intra_library_stats=True. Should contain similarities, not distances.
        Default: None.
    calculate_intra_library_stats : bool, optional
        Whether to calculate mean, median, and std of intra-library similarities.
        Requires library_similarity_matrix to be provided. Default: False.

    Returns
    -------
    Dict[str, Union[float, int, pd.DataFrame]]
        Dictionary containing:
        - Core metrics: n_targets, n_library, n_covered_targets, etc.
        - Coverage metrics: pct_covered_targets, pct_participating_library
        - Combined metrics: f1_coverage_participation, participation_adjusted_coverage_pct
        - Diversity metrics: evenness_of_participation, effective_participating_library
        - Diagnostics: avg_hits_per_covered_target
        - Intra-library metrics (if calculate_intra_library_stats=True):
          intra_library_similarity_mean, intra_library_similarity_median, 
          intra_library_similarity_std
    """
    # Accept DataFrame or ndarray; rows = targets, cols = library
    if isinstance(matrix, pd.DataFrame):
        target_ids = list(matrix.index)
        library_ids = list(matrix.columns)
        M = matrix.values
    else:
        M = np.asarray(matrix)
        target_ids = [f"T{i}" for i in range(M.shape[0])]
        library_ids = [f"L{j}" for j in range(M.shape[1])]

    # boolean hit matrix - use appropriate comparison operator based on matrix type
    if is_distance:
        H = M <= threshold  # for distances: smaller values indicate better hits
    else:
        H = M >= threshold  # for similarities: larger values indicate better hits
    n_targets, n_library = H.shape

    covered_mask = H.any(axis=1)
    participating_mask = H.any(axis=0)

    n_cov_targets = int(covered_mask.sum())
    n_participating = int(participating_mask.sum())

    C = n_cov_targets / n_targets if n_targets else 0.0
    P = n_participating / n_library if n_library else 0.0

    # Per-library hit counts (unique targets each library covers)
    hits_per_lib = H.sum(axis=0).astype(float)
    total_hits = hits_per_lib.sum()

    if n_participating > 0 and total_hits > 0:
        p = np.zeros(n_library, dtype=float)
        mask = hits_per_lib > 0
        p[mask] = hits_per_lib[mask] / total_hits
        H_shannon = -(p[mask] * np.log(p[mask])).sum()
        evenness = H_shannon / np.log(n_participating) if n_participating > 1 else 1.0
        n_eff = float(np.exp(H_shannon))
    else:
        evenness = 0.0
        n_eff = 0.0

    f1 = (2 * C * P) / (C + P) if (C + P) > 0 else 0.0
    pac = C * P * evenness

    # Redundancy diagnostics
    k_per_target = H.sum(axis=1)
    avg_hits_per_cov_target = (
        float(k_per_target[covered_mask].mean()) if n_cov_targets else 0.0
    )

    per_library = pd.DataFrame(
        {
            "participates": participating_mask,
            "targets_hit": hits_per_lib.astype(int),
            "hit_share": (hits_per_lib / total_hits) if total_hits else 0.0,
        },
        index=library_ids,
    ).sort_values("targets_hit", ascending=False)

    # Calculate intra-library similarity statistics if requested
    intra_lib_stats = {}
    if calculate_intra_library_stats:
        if library_similarity_matrix is None:
            raise ValueError(
                "library_similarity_matrix must be provided when calculate_intra_library_stats=True"
            )
        
        # Process library similarity matrix
        if isinstance(library_similarity_matrix, pd.DataFrame):
            lib_sim_matrix = library_similarity_matrix.values
        else:
            lib_sim_matrix = np.asarray(library_similarity_matrix)
        
        # Validate matrix is square
        if lib_sim_matrix.shape[0] != lib_sim_matrix.shape[1]:
            raise ValueError("library_similarity_matrix must be square (n_library x n_library)")
        
        # Validate matrix size matches library size
        if lib_sim_matrix.shape[0] != n_library:
            raise ValueError(
                f"library_similarity_matrix shape {lib_sim_matrix.shape} does not match "
                f"library size {n_library}"
            )
        
        # Filter similarity matrix to only participating compounds
        participating_lib_sim_matrix = lib_sim_matrix[participating_mask][:, participating_mask]
        
        # Extract upper triangle (excluding diagonal) to avoid double-counting pairs
        # and exclude self-similarity
        mask = np.triu(np.ones_like(participating_lib_sim_matrix, dtype=bool), k=1)
        intra_similarities = participating_lib_sim_matrix[mask]
        
        # Calculate statistics
        if len(intra_similarities) > 0:
            if is_distance:
                intra_similarities = 1 - intra_similarities
            intra_lib_stats = {
                "intra_library_similarity_mean": float(np.mean(intra_similarities)),
                "intra_library_similarity_median": float(np.median(intra_similarities)),
                "intra_library_similarity_std": float(np.std(intra_similarities, ddof=1)),
            }
        else:
            # Edge case: only one compound in library
            intra_lib_stats = {
                "intra_library_similarity_mean": 0.0,
                "intra_library_similarity_median": 0.0,
                "intra_library_similarity_std": 0.0,
            }

    result = {
        # Core
        "n_targets": n_targets,
        "n_library": n_library,
        "n_covered_targets": n_cov_targets,
        "pct_covered_targets": 100 * C,
        "n_participating_library": n_participating,
        "pct_participating_library": 100 * P,
        # Combined metrics
        "f1_coverage_participation": f1,
        "participation_adjusted_coverage_pct": 100 * pac,
        "evenness_of_participation": evenness,
        "effective_participating_library": n_eff,
        # Diagnostics
        "avg_hits_per_covered_target": avg_hits_per_cov_target,
    }
    
    # Add intra-library similarity statistics if calculated
    if intra_lib_stats:
        result.update(intra_lib_stats)
    
    return result


def coverage_metrics_multi_threshold(
    matrix: Union[pd.DataFrame, np.ndarray],
    thresholds: List[float],
    is_distance: bool = True,
    library_similarity_matrix: Optional[Union[pd.DataFrame, np.ndarray]] = None,
    calculate_intra_library_stats: bool = False,
) -> Dict[str, Dict[str, Union[float, int, pd.DataFrame]]]:
    """
    Compute coverage metrics across multiple thresholds.

    Parameters
    ----------
    matrix : pd.DataFrame or np.ndarray
        Distance or similarity matrix where rows = targets, cols = library compounds.
    thresholds : List[float]
        List of threshold values to evaluate.
    is_distance : bool, optional
        Whether the input matrix contains distances (True) or similarities (False).
        Default: False (similarity matrix).
    calculate_intra_library_stats : bool, optional
        Whether to calculate intra-library similarity statistics.
        Default: False.
    library_similarity_matrix : pd.DataFrame or np.ndarray, optional
        Library similarity matrix (n_library x n_library) for intra-library statistics.
        Required when calculate_intra_library_stats=True.

    Returns
    -------
    Dict[str, Dict[str, Union[float, int, pd.DataFrame]]]
        Dictionary with threshold values as keys, each containing full coverage metrics.
    """
    results = {}
    for threshold in thresholds:
        results[f"thresh_{threshold}"] = coverage_metrics(
            matrix, threshold, is_distance, library_similarity_matrix, calculate_intra_library_stats 
        )
    return results