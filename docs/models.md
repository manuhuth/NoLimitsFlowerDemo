# Models

A single run-config knob, `model`, selects one of four models. All four federate through the
same primitive — a per-subject sum — so the summed site contributions **are** the pooled
value and gradient exactly (**additivity**, checked to `1e-8` for every model). What differs
is the *acceptance* each model can support. The spread is deliberate: two classical PK
models, one **neural** mixed-effects model, and one **growth curve** — and the classical
warfarin PK model against a neural network on the *identical* warfarin data.

| `model` | data (all real) | kind | parameters | sites | acceptance |
|---|---|---|---|---|---|
| `warfarin` (default) | nlmixr2data warfarin, 32 subj / 251 obs | 1-cmt oral PK, closed-form ODE | 7 | 3 × 11/11/10 | strict: objective `1e-6`, params `1e-3` |
| `theophylline` | R `Theoph`, 12 subj / 132 obs | 1-cmt oral PK, closed-form ODE | 7 | 3 × 4 | strict: objective `1e-6`, params `1e-3` |
| `warfarin-nn` | same warfarin frame | **neural** mixed effects (FFNN mean) | 87 | 3 × 11/11/10 | **additivity gate** (`1e-8`) + reported objective agreement |
| `orange` | R `Orange`, 5 trees / 35 obs | logistic **growth** curve, algebraic | 5 | 2/2/1 | strict: objective `1e-6`, params `1e-3` |

Select a model at the command line:

```bash
flwr run . --stream --run-config 'model="warfarin"'      --federation-config ...   # default
flwr run . --stream --run-config 'model="theophylline"'  --federation-config ...
flwr run . --stream --run-config 'model="orange"'        --federation-config ...
flwr run . --stream --run-config 'model="warfarin-nn" max-rounds=40' --federation-config ...
```

The non-warfarin models default to and are documented for the `laplace` estimator.

## Data sources and licenses

The demo bundles small **real public** datasets so it is self-contained and offline. The
code is MIT; each **dataset keeps its own license** and is not covered by MIT.

| file | source | license | study |
|---|---|---|---|
| `data/warfarin.csv` | [`nlmixr2data::warfarin`](https://cran.r-project.org/package=nlmixr2data) (CRAN), committed verbatim | **GPL (≥ 3)** | O'Reilly single-dose oral warfarin PK/PD study |
| `data/theoph.csv` | R base `datasets::Theoph` | GPL-2 \| GPL-3 | theophylline PK (Upton; Boeckmann/Sheiner/Beal; Pinheiro & Bates) |
| `data/orange.csv` | R base `datasets::Orange` | GPL-2 \| GPL-3 | growth of orange trees (Draper & Smith; Pinheiro & Bates) |

All three are freely redistributable. No network is needed after checkout.

The warfarin frame is long PK/PD (515 rows, 32 subjects, `dvid` of `cp`/`pca`). The demo's
models are PK, so the loaders keep the plasma-concentration observations (`dvid == "cp"`,
`evid == 0`) and carry each subject's single dose (`amt`) as a constant covariate — **251
concentration observations from 32 subjects**, mapped to `ID=id`, `t=time`, `Dose=amt`
(60–153 mg) and `conc=dv`, split into 3 contiguous sites of 11/11/10.

A seeded synthetic warfarin set is available behind `data-source="simulated"` (24 subjects
at a known true `theta`, split 8/8/8). It powers the fast tests and the fault-injection test
and is the only mode with a known true `theta`.

## warfarin — the baseline

`model="warfarin"` is a population PK model: single oral dose, one compartment with
first-order absorption, and multiplicative log-normal random effects on absorption rate,
clearance and volume (`ka` 1/h, `cl` L/h, `v` L, concentrations mg/L).

```julia
@fixedEffects begin
    ka       = RealNumber(1.0)
    cl       = RealNumber(0.13)
    v        = RealNumber(8.0)
    omega_ka = RealNumber(0.4, scale=:log)
    omega_cl = RealNumber(0.3, scale=:log)
    omega_v  = RealNumber(0.2, scale=:log)
    sigma    = RealNumber(0.5, scale=:log)
end

@covariates begin
    t    = Covariate()
    Dose = ConstantCovariate(constant_on=:ID)
end

@randomEffects begin
    eta_ka = RandomEffect(LogNormal(0.0, omega_ka); column=:ID)
    eta_cl = RandomEffect(LogNormal(0.0, omega_cl); column=:ID)
    eta_v  = RandomEffect(LogNormal(0.0, omega_v);  column=:ID)
end

@preDifferentialEquation begin
    kai = ka * eta_ka
    cli = cl * eta_cl
    vi  = v * eta_v
end

@DifferentialEquation begin
    D(depot)   ~ -kai * depot
    D(central) ~ kai * depot - (cli / vi) * central
end

@initialDE begin
    depot   = Dose
    central = 0.0
end

@formulas begin
    cp = central(t) / vi
    conc ~ Normal(cp, sigma)
end
```

The ODE is linear in the states, so NoLimits takes its closed-form fast path; one site
objective-plus-gradient evaluation costs about 50 ms after compilation. The random effects
enter nonlinearly and the parameters mix scales (structural parameters plain, variance
parameters `scale=:log`) — a genuine ODE mixed-effects fit, not a linear toy.

**Acceptance rationale.** Fully identifiable, so the federated fit must match the pooled
`fit_model` on both objective (`1e-6`) and every natural-scale parameter (`1e-3`).

## theophylline

The same 1-compartment oral-absorption family as warfarin (depot → central, log-normal REs
on `ka`, `cl`, `v`), on R's real `Theoph` data (12 subjects). Closed-form ODE fast path,
same strict acceptance as warfarin.

## warfarin-nn — neural mixed effects on the *same* warfarin data

`warfarin-nn` replaces the PK structure with a feed-forward network on the **identical**
warfarin frame. The mean concentration is `NN([d, t, eta], nn_params)`, where `nn_params` is
an `FFNNParameters` block — a `(3, 5, 5, 5, 1)` tanh MLP, 86 weights — and `eta` is a
per-subject random effect. 87 parameters in total (`sigma` + 86 weights).

```julia
@Model begin
    @covariates begin
        t = Covariate()
        d = ConstantCovariate()
    end
    @fixedEffects begin
        sigma = RealNumber(1.0, scale=:log)
        nn_params = FFNNParameters((3, 5, 5, 5, 1); activation=:tanh, output_activation=:identity, function_name=:NN, calculate_se=false, seed=1234)
    end
    @randomEffects begin
        eta = RandomEffect(Normal(0.0, 1.0); column=:id)
    end
    @formulas begin
        mean_func = NN([d, t, eta], nn_params)[1]
        C ~ Normal(mean_func, sigma)
    end
end
```

### Why its acceptance is different — the additivity gate

The ~86 network weights are **non-identifiable**: permutation and sign symmetries of the
hidden units mean many distinct weight vectors give the same predictions and the same
likelihood. Two valid fits agree in objective and predictions while differing in weights, so
comparing parameters would be meaningless.

Federation is instead gated on the property federation is actually responsible for —
**additivity**: the sum over sites of `(value, gradient)` equals the pooled-data call at
`theta0` to `1e-8`. This is exact, and it is the headline claim for the neural model. The
full federated fit's objective is then compared to the pooled `fit_model` objective and
*reported*, not gated on parameters.

The `ServerApp` enforces this directly: when `acceptance == "nn"` it runs a self-contained
child probe, requires `value_rel < 1e-8` and `gradient_rel < 1e-8`, and logs
`PASS: NN site contributions are additive` before the fit even starts.

### Two seeds, and the demo-only warm start

- The `FFNNParameters` **`seed` is pinned in the model string** (`seed=1234`), so every
  site's Glorot-uniform weight initialization is identical. Without it, `theta0` would
  differ across sites and the prepare-round agreement check would abort the run — which is
  the *correct* behaviour: unpinned, the sites are not fitting the same model.
- Because the weights are non-identifiable, the federated fit is **warm-started from the
  pooled optimum** so the objective comparison is on the same basin. That warm-start is
  **demo-only**: a real deployment has no pooled dataset and would warm-start from a
  federated naive-pooled pass instead (naive-pooled is itself a per-subject sum, so it
  federates the same way).

!!! tip "The classical-vs-neural highlight"
    `warfarin` and `warfarin-nn` fit the **same 251 warfarin observations**: one with a
    mechanistic 1-compartment ODE (7 identifiable parameters, strict parameter-wise
    acceptance), the other with an 87-parameter neural mean function (non-identifiable
    weights, additivity gate). Federation is exact for **both** — the aggregation math does
    not care whether the mean function is a compartmental ODE or a neural network.

## orange — a growth curve, no ODE

The classic nlme Orange dataset: trunk circumference of 5 orange trees against age, fit with
a logistic growth curve and a per-tree random effect on the asymptote.

```julia
@Model begin
    @covariates begin
        age = Covariate()
    end
    @fixedEffects begin
        Asym  = RealNumber(200.0)
        xmid  = RealNumber(700.0)
        scal  = RealNumber(350.0)
        omega = RealNumber(50.0, scale=:log)
        sigma = RealNumber(10.0, scale=:log)
    end
    @randomEffects begin
        eta = RandomEffect(Normal(0.0, omega); column=:Tree)
    end
    @formulas begin
        circ = (Asym + eta) / (1.0 + exp((xmid - age) / scal))
        circumference ~ Normal(circ, sigma)
    end
end
```

It is non-PK, non-ODE and fully algebraic — the covariate `age` is referenced by name — so
it shows the federation math is not tied to PK or to ODEs. With only 5 trees the RE-variance
estimate `omega` is uncertain, but additivity is exact regardless of sample size and the fit
still matches the pooled `fit_model` within the strict tolerance.

## Equivalence, measured

**Additivity** on the real nlmixr2data warfarin data (3 sites of 11/11/10 subjects) at the
model's default `theta`. Summed site contributions vs the pooled-data call, per estimator:

| `estimator` | pooled objective | value rel. diff | gradient rel. diff |
|---|---|---|---|
| `laplace` | −657.2007051064 | 0.0 | 1.8e-16 |
| `focei` | −655.8320382738 | 1.7e-16 | 2.1e-16 |
| `ghq`, level 5 | −1131.1546377109 | 2.0e-16 | 2.3e-16 |
| `pooled` | −2570.1705666892 | 0.0 | 2.4e-16 |

The federated **fit** (`laplace`, default) vs the pooled `fit_model`, same start point,
passes the strict gate. Pooled `laplace` objective −455.7966, federated matching to
tolerance:

| parameter | value |
|---|---|
| ka | 0.5473 |
| cl | 0.1344 |
| v | 7.7002 |
| omega_ka | 0.4883 |
| omega_cl | 0.2837 |
| omega_v | 0.2200 |
| sigma | 1.0740 |

Sane single-dose oral warfarin PK values (clearance ~0.13 L/h, volume ~7.7 L). The residual
federated-vs-pooled parameter differences are optimizer tolerance, not federation error: the
site contributions themselves are exact (the additivity table above).
