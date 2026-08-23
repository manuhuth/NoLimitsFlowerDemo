# Optimizer & scipy interface

The server side of the fit is **pure numpy/scipy and never boots Julia**. It cannot: in the
Flower simulation runtime `@app.main` runs on a worker thread, and juliacall can only
cold-boot Julia from a process's main thread. Everything the optimizer needs — the parameter
names, the start point `theta0`, and the per-coordinate log mask — arrives from the sites in
the prepare round. Each objective/gradient evaluation is **one federated round**.

Source: `src/nolimits_flower/server_app.py`. The functions cited below are the ones to read
to follow the code.

## Non-DP: scipy L-BFGS-B, one round per evaluation

The fit is a single call to `scipy.optimize.minimize(method="L-BFGS-B", jac=True)` in
`_fit`. The objective closure is `federated(z)`, and each of its invocations broadcasts a
`theta` to every site and sums the returned `(value, gradient)`:

```python
# server_app._fit  (abridged)
s = task.precondition_scale(x0, mask)          # NoLimits' own preconditioning rule

def federated(z):
    x = x0 + s * np.asarray(z, dtype=float)    # preconditioned -> transformed theta
    sites = broadcast(grid, x, config, rnd=rounds)   # ONE federated round
    value = sum(v for _, v, _ in sites)              # sum site log-likelihoods
    grad  = np.sum([g for _, _, g in sites], axis=0) # sum site gradients
    return -value, -(s * grad)                 # L-BFGS-B minimizes; sites report a log-lik

res = minimize(
    federated, np.zeros_like(x0), method="L-BFGS-B", jac=True,
    options={"maxiter": max_rounds, "maxfun": max_rounds},
)
```

### How `jac=True` consumes the summed gradient

`jac=True` tells scipy the objective returns **both** the scalar and its gradient in one
call, as `(f, g)`. That is exactly the federated shape: one round produces one summed value
**and** one summed gradient, so there is no separate gradient round. `federated` returns
`(-value, -(s * grad))` — negated because the sites report a log-likelihood to be
**maximized** while L-BFGS-B **minimizes**, and scaled by `s` because the optimizer works in
the preconditioned coordinate `z` (chain rule: `grad_z = s * grad_theta`).

### One evaluation = one round

`broadcast` (`server_app.broadcast`) sends the current `theta` to every node via
`_send_all`, then returns `[(site_id, value, gradient)]`. `_send_all`
(`server_app._send_all`) builds one `Message` per node, calls `grid.send_and_receive`, and
**raises on any failure** — a missing reply, an error reply, or (checked in `broadcast`) a
non-finite contribution. A federated sum is only meaningful if every site is in it, so a
site failure aborts the whole fit rather than summing the survivors.

### `maxfun` is the round cap

`max-rounds` is passed as **`maxfun`**, the cap on function *evaluations* — i.e. federated
rounds. `maxiter` alone would not cap rounds: L-BFGS-B line searches spend extra evaluations
per iteration. A truncated run leaves `res.success` false and **fails the acceptance** rather
than passing off a half-optimized `theta` as the optimum.

### Preconditioning — replicated from NoLimits

The server optimizes a preconditioned coordinate `z` with `theta = theta0 + s * z`. The scale
`s` follows NoLimits' own rule, mirrored in `task.precondition_scale` from `_precondition_scale`
/ `_precondition_maps` in NoLimits.jl `src/estimation/common.jl`:

- `s_i = max(|theta0_i|, 1)` for a coordinate on the **identity** scale,
- `s_i = 1` for a **log-scaled** coordinate.

It is *reimplemented* in numpy rather than called, because it runs on the server, which has
no Julia. The per-coordinate log mask that distinguishes the two cases is reported by the
sites in the prepare round (`inverse_transform(0)` is 1 on a log-scaled coordinate and 0 on
an identity one), so `task.to_natural` and `task.precondition_scale` need no model-specific
bookkeeping.

**Why it matters:** the raw transformed coordinates mix a volume of ~8 (for `v`) with
unit-size log-parameters, which costs L-BFGS-B extra evaluations. On the warfarin model that
one number is worth **85 rounds down to 29**, and the raw scale also made L-BFGS-B exit with
the cosmetic `ABNORMAL` flag while the preconditioned one converges cleanly (`CONVERGENCE:
RELATIVE REDUCTION OF F <= FACTR*EPSMCH`). The exit flag is reported but nothing is gated on
it — the acceptance table is.

### The transformed scale crosses the wire

`theta` crosses the wire on the **transformed (unconstrained)** scale, so positivity
constraints stay implicit and the server needs no bounds. The client maps the wire vector
back to the natural scale with the model's own inverse transform inside Julia
(`task.objective_and_gradient` → the `nlf_objgrad` helper), and the returned gradient is on
the transformed axes — the scale the server optimizes on.

## DP: fixed-schedule Adam (designed)

Under differential privacy (see [Differential privacy](differential-privacy.md), a **planned
capability**), the optimizer changes. L-BFGS line searches break on noisy gradients and every
line-search evaluation would spend privacy budget, so DP mode uses a **fixed-schedule Adam**:

- a fixed learning rate `dp-lr`, no line search,
- a fixed number of rounds `dp-rounds` (`T`) — which *is* the privacy budget,
- **no convergence gating on noisy quantities** — the noisy objective and gradient norm are
  not trustworthy stopping signals, so the schedule runs to `T` and the accountant reports
  the spent `(ε, δ)`.

The round structure is otherwise identical: one broadcast, sum the (clipped, noised) site
gradients, one optimizer step.

## The scipy interface, concretely

The server is a small, self-contained scipy program:

- **Inputs** come from the prepare round: `prepare` (`server_app.prepare`) collects each
  site's reply and `agree` (`server_app.agree`) collapses them to the shared
  `(names, theta0, log_mask)` the sites must agree on — a disagreement means the sites are
  not running the same model, so the summed objective would be meaningless.
- **The objective** is `federated` (a closure inside `_fit`) returning `(f, g)` for
  `jac=True`.
- **The driver** is one `scipy.optimize.minimize(..., method="L-BFGS-B", jac=True)` call.
- **The output** is `theta_star = x0 + s * res.x`, re-broadcast once more for the per-site
  contributions, mapped to natural scale by `task.to_natural`, and checked against the pooled
  reference in the acceptance block.

No part of this imports or launches Julia. The only Julia-in-a-child-process on the server
side is the **demo-only** pooled reference fit (`server_app._child` → `python -m
nolimits_flower.task fit ...`), run *after* convergence purely to produce the acceptance
comparison; a production deployment deletes it.

!!! note "SecAgg wraps the sum"
    The two summations in `federated` — `sum(...)` over site values and `np.sum(...)` over
    site gradients — are exactly what secure aggregation replaces: with SecAgg the server
    receives only the *sum*, never the per-site payloads, and the optimizer code is unchanged.
    See [SecAgg (deployment only)](differential-privacy.md#secagg-deployment-only).
