import numpy as np
import nuclear as OLD

BALANCE_INTERVAL = 25
BALANCE_UNTIL = 100000
BALANCE_RATIO = 10.0
RHO_MIN, RHO_MAX = 0.125, 128.0


def rescale_duals(u, v, old_rho, new_rho):
    """Keep unscaled duals old_rho*u and old_rho*v invariant, in place."""
    scale = old_rho / new_rho
    u *= scale
    v *= scale


def balance_rho(rho, primal, dual):
    if primal > BALANCE_RATIO * dual:
        return min(RHO_MAX, 2 * rho)
    if dual > BALANCE_RATIO * primal:
        return max(RHO_MIN, rho / 2)
    return rho


def fit_nuclear(y, b, penalty=1.0, rho=2.0, max_iterations=200000,
                tolerance=1e-7, early_tolerance=1e-6, early_cap=10000,
                *, adaptive=True):
    """Return early_P, final_P, {early, final, rho_events, ...}.

Same positional API and objective as Stage A. rho is honored as the initial
rho; the evidence contract fixes it at 2. adaptive=False is for trajectory
parity testing only. All states start at zero. No hidden truth is accepted.
The checkpoint is the first both-residual early pass or early_cap, as in A.
"""
    OLD.validate_inputs(y, b)
    if not (0 < tolerance <= early_tolerance and 0 < early_cap <= max_iterations):
        raise ValueError("Invalid checkpoint/stopping settings")
    if penalty <= 0 or not (RHO_MIN <= rho <= RHO_MAX):
        raise ValueError("penalty must be positive and initial rho inside fixed bounds")
    n = len(b)
    initial_rho = rho
    inverse = np.linalg.inv(b @ b.T + 2 * rho * np.eye(n))
    data_term = y @ b.T
    p = np.zeros((n, n))
    lowrank, constrained, u, v = [np.zeros_like(p) for _ in range(4)]
    early_p, early_metadata = None, None
    converged, rho_events = False, []
    for iteration in range(1, max_iterations + 1):
        p = (data_term + rho * (lowrank - u + constrained - v)) @ inverse
        old_lowrank, old_constrained = lowrank, constrained
        left, singular, right = np.linalg.svd(p + u, full_matrices=False)
        lowrank = (left * np.maximum(singular - penalty / rho, 0)) @ right
        constrained = OLD.project_constraints(p + v)
        u += p - lowrank
        v += p - constrained
        primal = float(np.sqrt(np.sum((p - lowrank) ** 2) + np.sum((p - constrained) ** 2)))
        dual = float(rho * np.linalg.norm((lowrank - old_lowrank) + (constrained - old_constrained)))
        if not np.isfinite(primal + dual):
            raise FloatingPointError("Nonfinite ADMM residual")
        if early_p is None and ((primal < early_tolerance and dual < early_tolerance) or iteration == early_cap):
            early_p = constrained.copy()
            early_metadata = {"iterations": iteration,
                              "converged": bool(primal < early_tolerance and dual < early_tolerance),
                              "primal_residual": primal, "dual_residual": dual,
                              "objective": OLD.objective(early_p, y, b, penalty)}
        if primal < tolerance and dual < tolerance:
            converged = True
            break
        if (adaptive and iteration % BALANCE_INTERVAL == 0 and
                iteration <= BALANCE_UNTIL and iteration < max_iterations):
            new_rho = balance_rho(rho, primal, dual)
            if new_rho != rho:
                rho_events.append({"after_iteration": iteration, "old_rho": rho,
                                   "new_rho": new_rho, "dual_rescale": rho / new_rho,
                                   "primal_residual": primal, "dual_residual": dual})
                rescale_duals(u, v, rho, new_rho)
                rho = new_rho
                inverse = np.linalg.inv(b @ b.T + 2 * rho * np.eye(n))
    final_metadata = {"iterations": iteration, "converged": converged,
                      "primal_residual": primal, "dual_residual": dual,
                      "objective": OLD.objective(constrained, y, b, penalty)}
    assert early_p is not None
    return early_p, constrained, {"early": early_metadata, "final": final_metadata,
                                 "rho_events": rho_events, "initial_rho": initial_rho,
                                 "final_rho": rho, "adaptive": bool(adaptive),
                                 "rho_adaptation_last_permitted_iteration": BALANCE_UNTIL}
