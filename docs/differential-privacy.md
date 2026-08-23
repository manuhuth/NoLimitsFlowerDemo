# Differential privacy

!!! warning "Planned capability — designed, not yet wired into this showcase"
    Federated **is not private**: per-site log-likelihoods and gradients leak information
    about a site's subjects across rounds, so "no subject data leaves the site" is a
    *data-residency* claim, not a formal privacy guarantee. This page documents the
    **designed** differential-privacy layer — its guarantee, its knobs, and its accountant —
    which ships with the deployable version. The `dp*` and `secagg` knobs below are **not yet
    present in this showcase's `[tool.flwr.app.config]`**; the estimator, additivity and
    equivalence machinery on the other pages is what runs today. The `(σ, T) → ε` table is a
    genuine RDP-accountant computation for the designed mechanism, not a measurement from a
    DP run.

## What DP will guarantee

The target is **subject-level** differential privacy: adding or removing **one subject**
(one individual's entire longitudinal record) from a site changes the distribution of
everything the server sees only within a bounded factor, quantified as **(ε, δ)** and
tracked with a **Rényi DP (RDP)** accountant composed over the federated rounds.

This is the NLME analogue of DP-SGD. The federation primitive already exposes per-subject
gradient contributions (the batch form of `objective_and_gradient`), which is exactly the
unit DP needs to bound: one subject = one contribution to clip.

The mechanism, per round:

1. Each site computes its subjects' **per-subject gradient** contributions.
2. Each contribution is **clipped** to a fixed L2 norm `C` (`dp-clip`) — this bounds any one
   subject's influence, the sensitivity.
3. The clipped contributions are **summed**, and **Gaussian noise** of standard deviation
   `σ·C` (`dp-noise-multiplier`) is added to the sum.
4. The server takes one **fixed-schedule optimizer step** (Adam) on the noisy gradient.
5. The **RDP accountant** debits the round from the privacy budget; after `T` rounds
   (`dp-rounds`) the run reports the spent **(ε, δ)**.

The optimizer must change under DP: L-BFGS line searches break on noisy gradients and every
line-search evaluation would spend budget. So DP mode uses a **fixed-schedule Adam** with no
convergence gating on noisy quantities — see [Optimizer & scipy interface](optimizer.md).
`dp-rounds` is therefore both the iteration count **and** the privacy budget `T`.

## The knobs (designed)

All planned as `--run-config` overrides. Defaults are the designed defaults.

| key | default | meaning |
|---|---|---|
| `dp` | `false` | master switch: turn the DP mechanism on |
| `dp-clip` | `1.0` | per-subject gradient L2 clip norm `C` (the sensitivity bound) |
| `dp-noise-multiplier` | `1.0` | noise multiplier `σ`; Gaussian noise std added to the sum is `σ·C` |
| `dp-rounds` | `50` | number of DP-Adam rounds `T` — this **is** the privacy budget (composition length) |
| `dp-lr` | `0.05` | fixed Adam learning rate (no line search under noise) |
| `dp-delta` | `1e-5` | target `δ` for the `(ε, δ)` report from the accountant |
| `dp-clip-mode` | `joint` | `joint` (one clip over the whole gradient) or `per-group` (clip each parameter group separately) — see below |
| `dp-final-value` | `false` | also privately release the final objective value (noised) for reporting |
| `dp-value-clip` | `10.0` | clip on a subject's contribution to the objective value, when `dp-final-value` is on |

`secagg` is a separate switch documented under [SecAgg composition](#secagg-composition).

## Why `per-group` clipping exists — the ω collapse

This is the NLME-specific subtlety and the reason `dp-clip-mode` is a knob at all.

An NLME gradient mixes **very differently-scaled** blocks: structural parameters
(`ka`, `cl`, `v`), the **variance components** (`omega_*`), and the residual `sigma`. Under a
single **`joint`** clip, the large-norm structural block dominates the L2 norm, so the clip
scales the *whole* vector to keep the structural block in bounds — and the small variance-
component gradients get scaled toward zero along with it. Add isotropic Gaussian noise on top
and the variance components are all clip-suppressed signal plus full noise. The estimator
**collapses the omegas** — the RE variances shrink toward zero — which is a systematic bias,
not just added variance.

**`per-group`** clipping decouples the blocks: each parameter group gets its **own** clip
norm, so the variance-component gradients keep their signal instead of being scaled by the
structural block's magnitude. The key property is that this comes at **equal privacy cost**:
isotropic Gaussian noise calibrated to the **total** sensitivity `C_total` (the combined
per-group clips) gives the same RDP as a joint clip to `C_total`, so per-group buys
unbiased variance components without spending more budget.

!!! tip "Rule of thumb"
    Use `dp-clip-mode="per-group"` whenever the variance components (`omega_*`) matter —
    which for an NLME model is essentially always. `joint` is the simpler default and is fine
    when only the structural parameters are of interest.

## Reading ε from the results

When DP is on, the run's `results.json` will carry a `dp` block alongside the fit result.
Planned schema:

```json
{
  "dp": {
    "enabled": true,
    "clip": 1.0,
    "noise_multiplier": 4.0,
    "clip_mode": "per-group",
    "rounds": 50,
    "delta": 1e-5,
    "epsilon": 10.05,
    "accountant": "rdp-gaussian"
  }
}
```

- `epsilon` is the spent privacy budget at the chosen `delta`, from the RDP accountant over
  `rounds` composed Gaussian mechanisms.
- `noise_multiplier` and `rounds` are the two dials that move `epsilon`: more noise or fewer
  rounds → smaller `epsilon` (stronger privacy), at a utility cost.
- `clip` sets the sensitivity; with the standard normalization the accountant's `epsilon`
  depends on `noise_multiplier` and `rounds`, not on `clip` directly (clip trades bias for
  the noise-to-signal ratio, not the budget).

## Worked ε table — from the RDP accountant

The DP mode is **full-batch** (every subject participates every round; there is no Poisson
subsampling amplification), so the accounting is the composition of `T` Gaussian mechanisms.
For a Gaussian mechanism the RDP at order α is `α / (2σ²)` per round, `T·α / (2σ²)` composed,
converted to `(ε, δ)` by minimizing over α:

$$
\varepsilon(\delta) \;=\; \min_{\alpha > 1}\;
\frac{T\,\alpha}{2\sigma^2} \;+\; \frac{\log(1/\delta)}{\alpha - 1}.
$$

Spent **ε** at **δ = 1e-5**, for the full-batch Gaussian mechanism:

| `dp-noise-multiplier` (σ) | T = 25 | T = 50 | T = 100 |
|---|---|---|---|
| 0.5 | 97.99 | 167.86 | 295.97 |
| 1.0 | 36.49 | **58.93** | 97.99 |
| 2.0 | 15.12 | 23.22 | 36.49 |
| **4.0** | 6.78 | **10.05** | 15.12 |
| 8.0 | 3.19 | 4.63 | 6.78 |

Read it as the budget trade: at `T = 50` rounds, `σ = 1` spends **ε ≈ 59** (weak, essentially
data-residency plus a little noise), while `σ = 4` spends **ε ≈ 10** (a meaningful budget).
Halving the rounds or doubling the noise both tighten ε. These are exactly the values behind
the designed defaults: the mechanism is the standard full-batch Gaussian, and the accountant
is reproducible from the formula above.

!!! note "Honest utility caveat at small n"
    DP noise is calibrated to *one subject's* worst-case influence, but the demo's sites are
    tiny — 10–11 warfarin subjects, 4 theophylline, 1–2 orange trees. With so few subjects
    per site the clip removes a large fraction of the real signal and the noise dominates, so
    a *usefully small* ε (say ε ≤ 10) will visibly degrade the estimates, and the variance
    components suffer first (hence `per-group`). DP at this scale is a **correctness and
    accounting demonstration**, not a claim of good utility. Real deployments with hundreds
    to thousands of subjects per site are where a tight ε and usable estimates coexist.

## A full DP run (designed)

```bash
flwr run . --stream --run-config \
  'dp=true dp-clip=1.0 dp-noise-multiplier=4.0 dp-clip-mode="per-group" dp-rounds=50 dp-lr=0.05 dp-delta=1e-5' \
  --federation-config "num-supernodes=3 client-resources-num-cpus=3 init-args-num-cpus=3"
```

At `σ = 4`, `T = 50`, `δ = 1e-5` this targets **ε ≈ 10** with per-group clipping keeping the
variance components unbiased.

## SecAgg composition

Secure aggregation (SecAgg+, built into Flower) and DP compose and address **different**
threats:

- **SecAgg** hides the **per-site** contributions from the **server**: the server sees only
  the *sum* of the site payloads, never any one site's value or gradient, at **no accuracy
  cost**. The message protocol already keeps site payloads as plain records
  (`ArrayRecord` + `MetricRecord`), so the SecAgg mod wraps them without a redesign. A
  `secagg` run-config switch turns it on.
- **DP** bounds what the **sum itself** reveals about any one **subject**, via clipping and
  noise.

With both on, the noise can be added in a **distributed** way — each site adds its share of
the Gaussian noise before SecAgg sums the payloads — so:

- The server **never sees** any per-site value or gradient (SecAgg), only the noised sum.
- The server **cannot see** any individual noise share either; it observes only the final
  noised aggregate, which already carries the full `(ε, δ)` guarantee.
- No single party holds both a clean per-site gradient and the noise, which is what makes the
  guarantee hold against a curious server without a trusted aggregator.

## Roadmap position

The privacy hardening ships in this order, each reusing the same aggregation math:

1. **Secure aggregation** — server sees only the sum, no accuracy cost.
2. **Differential privacy** (this page) — per-subject clipping + Gaussian noise + RDP
   accountant, with the DP-Adam optimizer.
3. A **DataSHIELD (R) port** reusing the same aggregation math, whose disclosure filters are
   complementary to aggregate-level DP.
