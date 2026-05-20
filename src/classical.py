"""Classical recommender baselines (Popularity, Item-kNN CF, BPR).

Pure numpy/scipy — no GPU, no LLM. These run in a separate pass from the LLM
sweep (see scripts/run_classical.py) and produce a results JSON that's later
merged into the main results.json so plot_results.py overlays them on the
existing curves.

Items are identified by **title** (not item_id) to match what pair JSONLs
encode. The orchestrator builds a title->idx map from the same raw ratings
and min_reviews filter used by prepare_data.py, so every title appearing in
the pair files maps cleanly.

Score convention: higher is better. score_pair() uses strict > to match
src/evaluate.py:preference_accuracy semantics.
"""

import re

import numpy as np
from scipy import sparse


PAIR_RE = re.compile(r"^Approved:\s*(.*?),\s*Denied:\s*(.*)$")


def parse_pair_text(text: str) -> tuple[str, str]:
    """Return (liked_title, disliked_title) from a chosen/rejected string.

    Format guaranteed by src/pair_gen.py:105 — both chosen and rejected fields
    use 'Approved: X, Denied: Y' where X is always the liked item in chosen
    and the disliked item in rejected.
    """
    m = PAIR_RE.match(text.strip())
    if not m:
        raise ValueError(f"Unparseable pair text: {text!r}")
    return m.group(1).strip(), m.group(2).strip()


class BaseRecommender:
    def fit(self, bg_matrix, user_train_pairs, *, n_titles, seed):
        raise NotImplementedError

    def score(self, title_idx: int) -> float:
        raise NotImplementedError

    def score_pair(self, liked_idx: int, disliked_idx: int) -> bool:
        return float(self.score(liked_idx)) > float(self.score(disliked_idx))


class Popularity(BaseRecommender):
    """Per-title mean bg rating. Non-personalized; ignores user_train_pairs."""

    def fit(self, bg_matrix, user_train_pairs, *, n_titles, seed):
        bg = bg_matrix.tocsc()
        col_sum = np.asarray(bg.sum(axis=0)).flatten()
        col_nnz = np.diff(bg.indptr)
        scores = np.zeros(n_titles, dtype=np.float64)
        mask = col_nnz > 0
        scores[mask] = col_sum[mask] / col_nnz[mask]
        global_mean = float(scores[mask].mean()) if mask.any() else 0.0
        scores[~mask] = global_mean
        self._scores = scores

    def score(self, title_idx: int) -> float:
        return float(self._scores[title_idx])


class ItemKNN(BaseRecommender):
    """Item-item cosine similarity from background users.

    User profile: +1 for each liked occurrence in user_train_pairs, -1 for each
    disliked occurrence. score(t) = sum_j sim[t, j] * profile[j].

    The item-item sim matrix is cached on the class keyed by id(bg_matrix), so
    the orchestrator's one-time bg_matrix is reused across all (user, n, seed).
    """

    _sim_cache: dict = {}

    @classmethod
    def compute_sim(cls, bg_matrix) -> np.ndarray:
        key = id(bg_matrix)
        cached = cls._sim_cache.get(key)
        if cached is not None:
            return cached
        bg = bg_matrix.astype(np.float32)
        norms = np.sqrt(np.asarray(bg.multiply(bg).sum(axis=0)).flatten())
        gram_sp = (bg.T @ bg)
        gram = gram_sp.toarray().astype(np.float32)
        denom = np.outer(norms, norms)
        denom[denom == 0] = 1.0
        sim = gram / denom
        np.fill_diagonal(sim, 0.0)
        cls._sim_cache[key] = sim
        return sim

    @classmethod
    def clear_cache(cls):
        cls._sim_cache.clear()

    def fit(self, bg_matrix, user_train_pairs, *, n_titles, seed):
        sim = self.compute_sim(bg_matrix)
        profile = np.zeros(n_titles, dtype=np.float32)
        for liked, disliked in user_train_pairs:
            profile[liked] += 1.0
            profile[disliked] -= 1.0
        self._scores = sim @ profile

    def score(self, title_idx: int) -> float:
        return float(self._scores[title_idx])


class BPR(BaseRecommender):
    """Bayesian Personalized Ranking with frozen pretrained item embeddings.

    Phase 1 — pretrain (once per dataset/seed via BPR.pretrain):
        SGD on bg implicit positives (rating >= POS_RATING_THRESHOLD). Returns
        item embeddings W of shape (n_items, DIM).

    Phase 2 — per-user fit (called by orchestrator per (user, n, seed)):
        Freeze W (passed via constructor). Fit a fresh user vector u via SGD
        on user_train_pairs only, scoring score(t) = u @ W[t].
    """

    DIM = 32
    PRETRAIN_STEPS = 50_000
    USER_FIT_EPOCHS = 100
    LR = 0.05
    REG = 0.01
    POS_RATING_THRESHOLD = 4

    def __init__(self, W: np.ndarray | None = None):
        self._W = W
        self._u: np.ndarray | None = None

    @classmethod
    def pretrain(cls, bg_matrix, seed: int) -> np.ndarray:
        """Pretrain item embeddings on bg implicit positives. Pure SGD.

        Args:
            bg_matrix: scipy sparse, shape (n_bg_users, n_items). Values are ratings.
            seed: RNG seed (governs init, sampling order, and negative sampling).

        Returns:
            W: float32 ndarray of shape (n_items, DIM).
        """
        rng = np.random.default_rng(seed)
        bg = bg_matrix.tocoo()
        n_users, n_items = bg.shape

        pos_mask = bg.data >= cls.POS_RATING_THRESHOLD
        pos_users = bg.row[pos_mask]
        pos_items = bg.col[pos_mask]
        if len(pos_users) == 0:
            pos_users, pos_items = bg.row, bg.col
        n_pos = len(pos_users)

        if n_pos == 0:
            return rng.normal(0, 0.01, size=(n_items, cls.DIM)).astype(np.float32)

        user_pos_sets = [set() for _ in range(n_users)]
        for u, i in zip(pos_users, pos_items):
            user_pos_sets[u].add(int(i))

        H = rng.normal(0, 0.01, size=(n_users, cls.DIM)).astype(np.float32)
        W = rng.normal(0, 0.01, size=(n_items, cls.DIM)).astype(np.float32)

        lr = np.float32(cls.LR)
        reg = np.float32(cls.REG)

        n_steps = cls.PRETRAIN_STEPS
        sample_idx = rng.integers(0, n_pos, size=n_steps)
        neg_items = rng.integers(0, n_items, size=n_steps)

        for s in range(n_steps):
            k = int(sample_idx[s])
            u = int(pos_users[k])
            ip = int(pos_items[k])
            ineg = int(neg_items[s])
            retry = 0
            while ineg in user_pos_sets[u] and retry < 5:
                ineg = int(rng.integers(0, n_items))
                retry += 1

            diff = W[ip] - W[ineg]
            x = float(H[u] @ diff)
            sig = 1.0 / (1.0 + np.exp(-x))
            grad = np.float32(sig - 1.0)

            H_u_old = H[u].copy()
            H[u] -= lr * (grad * diff + reg * H[u])
            W[ip] -= lr * (grad * H_u_old + reg * W[ip])
            W[ineg] -= lr * (-grad * H_u_old + reg * W[ineg])

        return W

    def fit(self, bg_matrix, user_train_pairs, *, n_titles, seed):
        if self._W is None:
            raise RuntimeError(
                "BPR requires pretrained W. Call BPR.pretrain() and pass W to BPR(W=...)."
            )

        rng = np.random.default_rng(seed)
        u = rng.normal(0, 0.01, size=self.DIM).astype(np.float32)

        if user_train_pairs:
            lr = np.float32(self.LR)
            reg = np.float32(self.REG)
            n_pairs = len(user_train_pairs)
            order = rng.permutation(n_pairs * self.USER_FIT_EPOCHS) % n_pairs
            for k in order:
                liked, disliked = user_train_pairs[int(k)]
                diff = self._W[liked] - self._W[disliked]
                x = float(u @ diff)
                sig = 1.0 / (1.0 + np.exp(-x))
                grad = np.float32(sig - 1.0)
                u -= lr * (grad * diff + reg * u)

        self._u = u

    def score(self, title_idx: int) -> float:
        return float(self._u @ self._W[title_idx])
