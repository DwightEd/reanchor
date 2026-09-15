"""Finite-channel information and a constraint subspace, not hallucination scores.

All fitted targets are known selector variables from constructed relation pairs.
No natural hallucination annotation is consumed. A bits unit is not a novelty
claim; the sample space and the conditioning group are explicit below.
"""
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit, logsumexp, xlogy


def channel_information(channel):
    """I(C;J | one fixed context), with balanced C and categorical J.

    channel [2,K] includes the outside-source/output bucket. Never normalize
    away that bucket: a head that ignores both candidates must not look useful.
    """
    p = np.asarray(channel, float)
    if p.ndim != 2 or p.shape[0] != 2 or not np.isfinite(p).all() or (p < 0).any():
        raise ValueError('expected two finite nonnegative categorical rows')
    if not np.allclose(p.sum(1), 1, atol=2e-5):
        raise ValueError('channel rows must include all probability mass')
    p = p / p.sum(1, keepdims=True)
    mean = p.mean(0)
    logratio = np.log(np.maximum(p, 1e-300)) - np.log(np.maximum(mean, 1e-300))
    return float(np.sum(p * logratio) / (2 * np.log(2)))


def categorical(logits, candidates):
    """Unconditional candidate masses plus outside; candidates must be unique."""
    z = np.asarray(logits, float)
    ids = np.asarray(candidates, int)
    if z.ndim != 1 or len(set(ids.tolist())) != len(ids):
        raise ValueError('one logit row and distinct candidate IDs required')
    p = np.exp(z[ids] - logsumexp(z))
    return np.r_[p, max(0., 1 - p.sum())]


def row_basis(matrix, rank, atol=1e-8):
    x = np.asarray(matrix, float)
    if x.ndim != 2 or rank < 1 or not np.isfinite(x).all():
        raise ValueError('finite matrix and positive rank required')
    if not len(x) or np.linalg.norm(x) <= atol:
        return np.empty((0, x.shape[1]))
    _, s, v = np.linalg.svd(x, full_matrices=False)
    keep = min(rank, int(np.sum(s > max(atol, s[0]*1e-6))))
    return v[:keep]


def fit_constraint_space(selector_deltas, nuisance_deltas, rank=4, nuisance_rank=2):
    """SVD of paired selector changes after removing leading nuisance directions.

    Nuisances change record order or change BOTH query selector and record binding
    so the intended record ID stays fixed. This targets a semantic record-choice
    subspace, not a presumed raw ordinal ID; transfer is tested out of group.
    An empty space is unresolved, not silently replaced by a random direction.
    """
    d = np.asarray(selector_deltas, float)
    nuisance = row_basis(nuisance_deltas, nuisance_rank)
    clean = d - (d @ nuisance.T) @ nuisance
    basis = row_basis(clean, rank)
    return dict(basis=basis, nuisance=nuisance,
                selector_energy_before=float(np.sum(d*d)),
                selector_energy_after=float(np.sum(clean*clean)))


def projected_delta(delta, basis):
    d, b = np.asarray(delta, float), np.asarray(basis, float)
    if b.ndim != 2 or b.shape[1] != len(d):
        raise ValueError('subspace/delta dimension mismatch')
    return (d @ b.T) @ b


def address_components(delta_q, rotated_keys):
    """Exact address-discriminating subspace at fixed K (query already rotated).

    Relative logits depend ONLY on span{k_j-k_0}; its orthogonal complement can
    still change attention to unlisted tokens, so it need not leave output fixed.
    This is an algebraic control, NOT an empirical mechanism discovery.
    """
    k = np.asarray(rotated_keys, float)
    basis = row_basis(k[1:] - k[:1], max(1, len(k)-1))
    parallel = projected_delta(delta_q, basis)
    return parallel, np.asarray(delta_q) - parallel


def contrast(q, keys, scale):
    k = np.asarray(keys, float)
    if k.shape[0] != 2:
        raise ValueError('the exact log-odds identity uses two anchor positions')
    return float(np.dot(q, k[1]-k[0])*scale)


def decompose_qk(q0, q1, k0, k1, scale):
    """Lambda_11-Lambda_00 = Q-only + K-only + bilinear interaction."""
    d0, d1 = k0[1]-k0[0], k1[1]-k1[0]
    dq, dk = q1-q0, d1-d0
    return dict(q_effect=float(dq@d0*scale), k_effect=float(q0@dk*scale),
                interaction=float(dq@dk*scale),
                total=float((q1@d1-q0@d0)*scale))


def fit_probe(x, labels, rank=32, penalty=1.):
    """Small frozen semantic decoder. The LLM and routing algorithm are untrained.

    PCA and logistic weights are reference-only; no correct/error filtering.
    This decoder is a measurement instrument, not an unsupervised detector.
    """
    x, y = np.asarray(x, float), np.asarray(labels, float)
    if x.ndim != 2 or len(x) != len(y) or set(y) != {0., 1.}:
        raise ValueError('probe requires both balanced selector conditions')
    center = x.mean(0)
    basis = row_basis(x-center, rank)
    if not len(basis):
        return dict(center=center, basis=basis, scale=np.ones(0), weight=np.zeros(1))
    z = (x-center) @ basis.T
    scale = np.maximum(z.std(0), 1e-6)
    z = np.column_stack((z/scale, np.ones(len(z))))
    def objective(w):
        logits = z@w
        loss = np.mean(np.logaddexp(0, logits)-y*logits) + .5*penalty*np.dot(w[:-1], w[:-1])/len(y)
        grad = z.T@(expit(logits)-y)/len(y)
        grad[:-1] += penalty*w[:-1]/len(y)
        return float(loss), grad
    fitted = minimize(objective, np.zeros(z.shape[1]), jac=True, method='L-BFGS-B',
                      options={'maxiter':500, 'ftol':1e-10})
    if not fitted.success:
        raise RuntimeError('probe optimization failed: '+str(fitted.message))
    return dict(center=center, basis=basis, scale=scale, weight=fitted.x)


def probe_logits(probe, x):
    z = (np.asarray(x)-probe['center'])@probe['basis'].T/probe['scale']
    return np.column_stack((z, np.ones(len(z))))@probe['weight']


def accessible_bits(logits, y):
    """1 - held-out cross entropy, for a balanced binary intervention variable.

    A population variational lower bound, empirically estimated here. May be
    negative. NOT exact hidden MI, NOT Fano inversion, NOT a per-token bit count.
    """
    y, z = np.asarray(y), np.asarray(logits)
    return float(1-np.mean(np.logaddexp(0, z)-y*z)/np.log(2))


def cluster_interval(values, seed=20260915, draws=200):
    x = np.asarray(values, float)
    if not len(x):
        return dict(n_groups=0, mean=None, ci95=None)
    result = dict(n_groups=len(x), mean=float(x.mean()), ci95=None)
    if len(x)>1 and draws:
        rng = np.random.default_rng(seed)
        means = np.mean(x[rng.integers(len(x), size=(draws, len(x)))], axis=1)
        result['ci95'] = np.quantile(means,[.025,.975]).tolist()
    return result


def receiver_coordinates(base,donor,site,layer,head):
    """Compare different-length contexts at the SAME receiver rotary positions."""
    def rotate(x,cos,sin):
        a,b=np.split(np.asarray(x,float),2,axis=-1)
        return x*cos+np.concatenate((-b,a),axis=-1)*sin
    q=rotate(donor['q_raw'][site,layer,head],base['query_cos'][site],base['query_sin'][site])
    k=rotate(donor['k_raw'][site,layer,head],base['anchor_cos'],base['anchor_sin'])
    return q,k
