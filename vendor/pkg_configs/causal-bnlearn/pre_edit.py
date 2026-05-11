"""Pre-edit for causal-bnlearn: patch local_score_BDeu to fix pandas 2.x compatibility.

The original implementation uses pandas groupby.get_group() which breaks with
numpy integer keys in pandas 2.x. This replaces the pandas-heavy implementation
with a pure numpy version that is both faster and compatible.
"""

_BDEU_REPLACEMENT = '''def local_score_BDeu(Data: ndarray, i: int, PAi: List[int], parameters=None) -> float:
    """
    Calculate the *negative* local score with BDeu for the discrete case.

    Parameters
    ----------
    Data: (sample, features)
    i: current index
    PAi: parent indexes
    parameters:
                 sample_prior: sample prior
                 structure_prior: structure prior
                 r_i_map: number of states of the finite random variable X_{i}

    Returns
    -------
    score: local BDeu score
    """
    if parameters is None:
        sample_prior = 1
        structure_prior = 1
        r_i_map = {
            idx: len(np.unique(np.asarray(Data[:, idx]))) for idx in range(Data.shape[1])
        }
    else:
        sample_prior = parameters["sample_prior"]
        structure_prior = parameters["structure_prior"]
        r_i_map = parameters["r_i_map"]

    r_i = r_i_map[i]
    q_i = 1
    for pa in PAi:
        q_i *= r_i_map[pa]

    N = Data.shape[0]
    vm = Data.shape[1] - 1

    # Structure prior
    BDeu_score = len(PAi) * np.log(structure_prior / vm) + (vm - len(PAi)) * np.log(
        1 - (structure_prior / vm)
    )

    alpha_ij = sample_prior / q_i
    alpha_ijk = sample_prior / (r_i * q_i)

    if len(PAi) == 0:
        # No parents: single parent config
        Nij = N
        # Count occurrences of each value of X_i
        vals, counts = np.unique(Data[:, i].astype(int), return_counts=True)
        first_term = math.lgamma(alpha_ij) - math.lgamma(Nij + alpha_ij)
        second_term = 0.0
        for c in counts:
            second_term += math.lgamma(c + alpha_ijk) - math.lgamma(alpha_ijk)
        # Add contribution from unobserved states of X_i
        n_unobserved = r_i - len(vals)
        if n_unobserved > 0:
            second_term += n_unobserved * (math.lgamma(alpha_ijk) - math.lgamma(alpha_ijk))
        BDeu_score += first_term + second_term
    else:
        # Encode parent configurations as a single integer
        pa_data = Data[:, PAi].astype(int)
        # Compute a unique key for each parent configuration
        multipliers = np.ones(len(PAi), dtype=np.int64)
        for idx in range(len(PAi) - 1, 0, -1):
            multipliers[idx - 1] = multipliers[idx] * r_i_map[PAi[idx]]
        pa_keys = pa_data @ multipliers

        child_data = Data[:, i].astype(int)

        # Get unique parent configs and their counts
        unique_pa, inverse, pa_counts = np.unique(pa_keys, return_inverse=True, return_counts=True)

        for g_idx in range(len(unique_pa)):
            Nij = pa_counts[g_idx]
            mask = inverse == g_idx
            vals, counts = np.unique(child_data[mask], return_counts=True)

            first_term = math.lgamma(alpha_ij) - math.lgamma(Nij + alpha_ij)
            second_term = 0.0
            for c in counts:
                second_term += math.lgamma(c + alpha_ijk) - math.lgamma(alpha_ijk)
            BDeu_score += first_term + second_term

        # Unobserved parent configs contribute only prior terms
        n_unobserved_pa = q_i - len(unique_pa)
        if n_unobserved_pa > 0:
            prior_term = math.lgamma(alpha_ij) - math.lgamma(alpha_ij)
            prior_child = r_i * (math.lgamma(alpha_ijk) - math.lgamma(alpha_ijk))
            BDeu_score += n_unobserved_pa * (prior_term + prior_child)

    return -BDeu_score
'''

OPS = [
    {
        "op": "replace",
        "file": "causal-bnlearn/causallearn/score/LocalScoreFunction.py",
        "start_line": 78,
        "end_line": 172,
        "content": _BDEU_REPLACEMENT,
    },
]
