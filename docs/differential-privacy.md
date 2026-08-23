# Differential privacy

!!! success "As-built — DP runs in this showcase's simulation today"
    Setting `dp=true` switches the federated fit from L-BFGS-B to a fixed-schedule
    **DP-Adam** loop over per-subject-clipped, Gaussian-noised gradients, and writes the
    spent **(ε, δ)** to `results.json`. The `dp*` knobs below are live in
    `[tool.flwr.app.config]` and every value in the ε table is the run's own RDP accountant.
    **SecAgg** is the one privacy feature that stays deployment-only (see
    [SecAgg](#secagg-deployment-only)): it needs the legacy deployment runtime, and this
    showcase uses the Message-API simulation.

Federated learning alone **is not private**: per-site log-likelihoods and gradients leak
information about a site's subjects across rounds, so "no subject data leaves the site" is a
*data-residency* claim, not a formal privacy guarantee. This page documents the
differential-privacy layer that closes that gap for the aggregate release.

## What DP guarantees

The target is **subject-level** differential privacy: adding or removing **one subject**
(one individual's entire longitudinal record) from a site changes the distribution of
everything the server sees only within a bounded factor, quantified as **(ε, δ)** and
tracked with a **Rényi DP (RDP)** accountant composed over the federated rounds.

This is the NLME analogue of DP-SGD. The federation primitive already exposes per-subject
gradient contributions (the batch form of `objective_and_gradient`, `task.nlf_dp_batches`),
which is exactly the unit DP needs to bound: one subject = one contribution to clip. (When a
random-effect batch spans more than one subject the clipping unit becomes that batch and the
run says so; for this demo's one-ID-grouped-RE models a batch is one subject.)

The mechanism, per round:

1. Each site computes its subjects' **per-subject gradient** contributions, on the
   preconditioned coordinate the server's optimizer steps in.
2. Each contribution is **clipped** to a fixed L2 norm `C` (`dp-clip`) — this bounds any one
   subject's influence, the sensitivity.
3. The clipped contributions are **summed**, and each of the `S` sites adds its **share** of
   Gaussian noise, `N(0, (σ·C)² / S)` per coordinate, so the noise on the *federated sum* is
   exactly `N(0, (σ·C)²)` — the Gaussian mechanism at noise multiplier `σ`
   (`dp-noise-multiplier`). The noise is **distributed** so it composes with SecAgg in
   deployment: no party ever holds a pre-noise sum.
4. The server takes one **fixed-schedule Adam** step on the noisy gradient.
5. The **RDP accountant** debits the round; after `T` rounds (`dp-rounds`) the run reports the
   spent **(ε, δ)** in `results.json`.

The optimizer must change under DP: L-BFGS line searches break on noisy gradients and every
line-search evaluation would spend budget. So DP mode uses **fixed-schedule Adam** with no
convergence gating on noisy quantities — see [Optimizer & scipy interface](optimizer.md).
`dp-rounds` is therefore both the iteration count **and** the privacy budget `T`.

`estimator="pooled"` is **rejected under DP**: the naive-pooled objective calibrates its
plug-in random effects on the whole data set and has no per-subject form, so no per-subject
clipping bound exists for it. Use `laplace`, `focei` or `ghq`.

## The knobs

All are `--run-config` overrides; defaults are the defaults in `[tool.flwr.app.config]`.

| key | default | meaning |
|---|---|---|
| `dp` | `false` | master switch: turn the DP mechanism on |
| `dp-clip` | `20.0` | per-subject gradient L2 clip norm `C` (the sensitivity bound); a **public** hyperparameter tuned to the model — see [Choosing `dp-clip`](#choosing-dp-clip) |
| `dp-noise-multiplier` | `1.0` | noise multiplier `σ`; Gaussian noise std added to the sum is `σ·C` |
| `dp-rounds` | `50` | number of DP-Adam rounds `T` — this **is** the privacy budget (composition length) |
| `dp-lr` | `0.05` | fixed Adam learning rate (no line search under noise) |
| `dp-delta` | `1e-5` | target `δ` for the `(ε, δ)` report from the accountant |
| `dp-clip-mode` | `per-group` | `per-group` (clip each parameter group separately — the default, preserves the variance components) or `joint` (one clip over the whole gradient) — see below |
| `dp-final-value` | `false` | also privately release the final objective value (noised) for reporting |
| `dp-value-clip` | `100.0` | clip on a subject's contribution to the objective value, when `dp-final-value` is on |
| `dp-groups` | `""` | `per-group` only: `name:group,...` overrides for the group classifier |
| `dp-clip-per-group` | `""` | `per-group` only: `group:clip,...` per-group clip norms `C_g` |
| `results-path` | `results.json` | where the DP fit writes its result and the `dp` block |

## Why `per-group` clipping exists — the ω collapse

This is the NLME-specific subtlety and the reason `dp-clip-mode` is a knob at all.

An NLME gradient mixes **very differently-scaled** blocks: structural parameters
(`ka`, `cl`, `v`), the **variance components** (`omega_*`), and the residual `sigma`. Under a
single **`joint`** clip, the large-norm structural block dominates the L2 norm, so the clip
scales the *whole* vector to keep the structural block in bounds — and the small variance-
component gradients get scaled toward zero along with it. Add isotropic Gaussian noise on top
and the variance components are all clip-suppressed signal plus full noise. The estimator
**collapses the omegas** — the RE variances shrink toward zero — a systematic bias, not just
added variance.

**`per-group`** clipping decouples the blocks. The classifier
(`task.dp_param_group`) sorts each coordinate into a **`variance`** group (names matching
`omega`, `sigma`, `tau`, `cov`, `sd`, `var`, `corr`, `rho`) or a **`location`** group
(everything else); `dp-groups` overrides a misclassification, `dp-clip-per-group` sets each
group's own clip `C_g`. Each subject's per-group sub-vector is clipped to `C_g`
independently, so the variance-component gradients keep their signal instead of being scaled
by the structural block's magnitude.

The key property is **equal privacy cost**. Clipping subject *i*'s group-*g* sub-vector to
`C_g` bounds its whole concatenated contribution by

$$
C_\text{total} \;=\; \sqrt{\textstyle\sum_g C_g^2},
$$

so add/remove-one-subject moves the site sum by at most `C_total` in L2 — `C_total` **is** the
sensitivity of the concatenated release. We then add **isotropic** noise of standard deviation
`σ·C_total` on *every* coordinate (not `σ·C_g` per group). The release is one Gaussian
mechanism with sensitivity `C_total` and noise `σ·C_total`, i.e. noise multiplier `σ` —
identical accounting to a joint clip at `C_total`. So **ε is unchanged** whichever clip mode is
in force: it depends only on `σ`, the round count and `δ`, never on the clip or the split.
Per-group buys unbiased variance components at **no extra budget**. (Noising per group at
`σ·C_g` instead would give a block-scaled Gaussian whose per-round RDP is `G` times worse, so
the demo does **not** do that.)

!!! tip "Rule of thumb"
    Keep the default `dp-clip-mode="per-group"` whenever the variance components (`omega_*`)
    matter — which for an NLME model is essentially always. `joint` collapses the omegas out of
    the box and is only worth choosing when *only* the structural parameters are of interest.

## Choosing `dp-clip`

`dp-clip` is a **public** hyperparameter — a fixed number you set and tune, exactly like
`dp-lr`. It is **not** learned and must **never** be read off the private gradients or data:
choosing `C` from the private per-subject norms would itself leak information and break the
`(ε, δ)` guarantee. Pick it from public knowledge of the model's scale, then leave it fixed.

The right size is the model's typical **per-subject gradient magnitude** on the preconditioned
coordinate the optimizer steps in:

- **Too small** → every subject's gradient saturates the clip, so the clip scales all of them
  down toward zero; the small variance-component signal is lost first and the **omegas
  collapse** even before noise is added. (This is what the old `dp-clip=1.0` default did to
  warfarin, whose per-subject gradient norms run ~20+.)
- **Too large** → the sensitivity `C` is bigger than it needs to be, so at a fixed `σ` the
  Gaussian noise `σ·C` is larger than necessary and utility suffers.

The default `dp-clip=20.0` suits this demo's **warfarin** model (the default `model`). Other
models sit at different scales — a differently-parameterised or differently-preconditioned
model may want a larger or smaller `C` — so retune it when you change the model. Combine it with
the default `per-group` mode: per-group keeps each block's signal even when a single `C` is not
a perfect fit for every block.

## A full DP run

```bash
cd NoLimitsFlowerDemo
flwr run . --stream --run-config \
  'model="theophylline" dp=true dp-noise-multiplier=4.0 dp-clip-mode="per-group" dp-rounds=50 dp-lr=0.05 dp-delta=1e-5' \
  --federation-config "num-supernodes=3 client-resources-num-cpus=3 init-args-num-cpus=3"
```

At `σ = 4`, `T = 50`, `δ = 1e-5` this targets **ε ≈ 10** with per-group clipping keeping the
variance components unbiased. The equal-CPU federation config pins the Ray actor pool to one
actor so each site's DataModel is built once, in the prepare round (see
[Architecture](architecture.md)).

## Reading ε from the results

DP writes a `results.json` (path from `results-path`) with a `dp` block alongside the fit.
The `theta_natural` / `theta_transformed` fields are the DP-Adam optimum; nothing un-noised is
reported — no per-site contribution, no objective trajectory, and no objective unless
`dp-final-value` asked for one. A real block from the **`theophylline`** model (note the
theophylline scale — `v ≈ 0.46`, `cl ≈ 0.039`; a different model's numbers live at a different
scale, so read the values against the model that produced them, not against warfarin):

```json
{
  "model": "theophylline",
  "theta_natural": { "ka": 1.53, "cl": 0.039, "v": 0.46, "omega_ka": 0.51, "...": "..." },
  "dp": {
    "enabled": true,
    "adjacency": "add/remove one subject",
    "unit": "subject",
    "epsilon": 143.46,
    "delta": 1e-05,
    "releases": 41,
    "sites": 3,
    "clip-mode": "joint",
    "noise": "distributed: each site adds N(0, (sigma*clip)^2 / sites)",
    "noise-multiplier": 0.5,
    "rounds": 40,
    "epsilon-per-site-vs-server": 352.45
  }
}
```

- `epsilon` is the spent budget at `delta`, from the RDP accountant over `releases` composed
  Gaussian mechanisms (`dp-rounds`, plus one if `dp-final-value` released the objective).
- `noise-multiplier` and `rounds` are the two dials that move `epsilon`: more noise or fewer
  rounds → smaller `epsilon` (stronger privacy), at a utility cost. `clip` does **not** enter
  `epsilon` (it trades bias for the noise-to-signal ratio, not the budget).
- `epsilon-per-site-vs-server` is the honest small print of running DP **without** SecAgg in
  this simulation: the server also sees each site's *own* noised release, which carries only
  its `1/S` share of the noise, so against the server a single site's subjects spend a budget
  `√S` larger. SecAgg (deployment only) closes this — the server would then see only the sum.
- Under `per-group` the block also carries `groups`, `group-clips` and `clip-total`.

## Worked ε table — from the RDP accountant

DP mode is **full-batch** (every subject participates every round; no Poisson subsampling
amplification), so the accounting is the composition of `T` Gaussian mechanisms. The RDP at
order α is `α / (2σ²)` per round, `T·α / (2σ²)` composed, converted to `(ε, δ)` by minimizing
over α:

$$
\varepsilon(\delta) \;=\; \min_{\alpha > 1}\;
\frac{T\,\alpha}{2\sigma^2} \;+\; \frac{\log(1/\delta)}{\alpha - 1}.
$$

Spent **ε** at **δ = 1e-5**, straight from `task.dp_epsilon` (the same code the run reports):

| `dp-noise-multiplier` (σ) | T = 25 | T = 50 | T = 100 |
|---|---|---|---|
| 0.5 | 97.99 | 167.86 | 295.97 |
| 1.0 | 36.49 | **58.93** | 97.99 |
| 2.0 | 15.12 | 23.22 | 36.49 |
| **4.0** | 6.78 | **10.05** | 15.12 |
| 8.0 | 3.19 | 4.63 | 6.78 |

Read it as the budget trade: at `T = 50`, `σ = 1` spends **ε ≈ 59** (weak, essentially
data-residency plus a little noise), while `σ = 4` spends **ε ≈ 10** (a meaningful budget).
Halving the rounds or doubling the noise both tighten ε. Because per-group clipping has the
same accounting as joint at `C_total`, these values are exactly the ε both clip modes report.

!!! note "Honest utility caveat at small n"
    DP noise is calibrated to *one subject's* worst-case influence, but the demo's sites are
    tiny — 4 theophylline subjects per site, 1–2 orange trees, 8 warfarin. With so few
    subjects per site the clip removes a large fraction of the real signal and the noise
    dominates, so utility degrades as ε tightens. With the defaults tuned for the model (a
    `dp-clip` sized to its per-subject gradient magnitude and `per-group` clipping) a fit stays
    **decent down to about ε ≈ 10 and gets poor below it**; the variance components suffer
    first, which is exactly why `per-group` is the default. **DP at this scale is a correctness
    and accounting demonstration, not a claim of good utility** — good utility needs a weaker ε
    or larger cohorts. Real deployments with hundreds to thousands of subjects per site are
    where a tight ε and usable estimates coexist. This is why the DP path does not gate on the
    pooled-fit acceptance the non-DP path uses: the noised optimum is not meant to match the
    exact pooled fit.

## SecAgg — deployment only

Secure aggregation (SecAgg+, built into Flower) and DP compose and address **different**
threats:

- **SecAgg** hides the **per-site** contributions from the **server**: the server sees only
  the *sum* of the site payloads, never any one site's value or gradient, at **no accuracy
  cost**.
- **DP** bounds what the **sum itself** reveals about any one **subject**, via clipping and
  noise. This is what runs in the showcase.

With both on, the noise is added in a **distributed** way — each site adds its share before
SecAgg sums the payloads — so the server never sees any per-site value, never sees any
individual noise share, and observes only the final noised aggregate, which already carries
the full `(ε, δ)` guarantee. No single party holds both a clean per-site gradient and the
noise, which is what makes the guarantee hold against a curious server without a trusted
aggregator. That is why the DP noise here is already distributed (`(σ·C)² / S` per site): the
mechanism is SecAgg-ready.

!!! warning "SecAgg is not wired into this showcase (coming soon)"
    SecAgg+ runs through Flower's `secaggplus_mod`, which masks a legacy `FitRes`-shaped
    **train** reply — it belongs to the **deployment** runtime (one SuperNode process per
    site). This showcase runs the **Message-API simulation**, where the sites answer plain
    `@app.query` messages, so the SecAgg mod has nothing to wrap. The deployable version of
    this app adds the `secagg` switch and the masked round; DP, which is pure clipping + noise
    on the payload, needs none of that and runs here today. The `epsilon-per-site-vs-server`
    field above quantifies exactly what SecAgg would remove.

## Roadmap position

1. **Differential privacy** (this page) — per-subject clipping + Gaussian noise + RDP
   accountant with the DP-Adam optimizer. **Runs in the simulation now.**
2. **Secure aggregation** — server sees only the sum, no accuracy cost. Deployment runtime.
3. A **DataSHIELD (R) port** reusing the same aggregation math, whose disclosure filters are
   complementary to aggregate-level DP.
