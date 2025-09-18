# Model Card: LTV and Uplift

## Overview
- LTV: BG/NBD for purchase frequency + Gamma-Gamma for spend per transaction.
- Uplift: Two-Model logistic classifiers for treatment and control.

## Intended Use
- Plan retention and growth investments; forecast revenue; design experiments; target promotions to maximally improve net revenue.

## Data
- Synthetic users/events/orders with seasonality, heterogeneous propensities, and a 28-day experiment with modest uplift concentrated in persuadable users.

## Metrics
- LTV: WMAPE on 60-day holdout; calibration plot.
- Uplift: Qini/AUUC; simulator for net revenue lift.

## Ethical/Operational
- Avoid targeting vulnerable populations; respect fairness constraints.
- Ensure promotions do not degrade long-term LTV.

## Limitations
- Uplift limited to Two-Model; does not account for interference.
- LTV assumes stationarity; segments may drift.
