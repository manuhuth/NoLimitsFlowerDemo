# Bundled datasets and their licenses

The demo code in this repository is MIT licensed. The datasets bundled here are **not**
covered by that MIT license: each keeps the license of its upstream source, documented
below. Redistribution of all three is permitted; only the licenses differ.

## warfarin.csv

- **Source:** `nlmixr2data::warfarin` from the CRAN package
  [nlmixr2data](https://cran.r-project.org/package=nlmixr2data), committed verbatim.
- **License:** GPL (>= 3) (inherited from nlmixr2data). This GPL-licensed data file is
  bundled inside an otherwise-MIT repository; the data file carries its own license.
- **Study/citation:** the O'Reilly warfarin single-dose PK/PD study. O'Reilly RA,
  Aggeler PM, Leong LS (1963/1964), *Studies on the coumarin anticoagulant drugs*,
  J. Clin. Invest. The dataset was popularized in pharmacometrics via Monolix/NONMEM
  and is distributed under GPL by nlmixr2data.
- **Format:** long PK/PD, 515 rows, 32 subjects. Columns `id, time, amt, dv, dvid,
  evid, wt, age, sex`, where `dvid` is `cp` (plasma warfarin concentration) or `pca`
  (prothrombin complex activity, the PD response). The demo uses the PK arm only:
  the loaders keep `dvid == "cp"` observations (`evid == 0`) and carry each subject's
  single dose (`amt` on its `evid == 1` row) as a constant covariate.

## theoph.csv

- **Source:** the `Theoph` dataset from R's base `datasets` package.
- **License:** GPL-2 | GPL-3 (R base). Freely redistributable (as catalogued by
  Rdatasets).
- **Study/citation:** theophylline PK. Data of Dr. Robert Upton; documented by
  Boeckmann, Sheiner & Beal (NONMEM), popularized by Pinheiro & Bates,
  *Mixed-Effects Models in S and S-PLUS* (2000).
- **Format:** 12 subjects. Columns `Subject, Wt, Dose, Time, conc`.

## orange.csv

- **Source:** the `Orange` dataset from R's base `datasets` package.
- **License:** GPL-2 | GPL-3 (R base). Freely redistributable.
- **Study/citation:** growth of orange trees. Draper & Smith, *Applied Regression
  Analysis*; used as an nlme example by Pinheiro & Bates (2000).
- **Format:** 5 trees. Columns `Tree, age, circumference`.
