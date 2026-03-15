# Research Workflow Architecture

## Goals

The upgraded engine is designed to be:

- explainable
- timestamp-safe
- auditable
- adaptable under governance
- local-first and research-friendly

## Key Design Choices

### 1. Separate user-facing suggestions from research artifacts

`Suggestion` is the human-readable output.

`SignalSnapshot` is the persisted research artifact used for later evaluation and adaptation. It stores:

- signal timestamp
- commodity and contract
- regime label
- direction and conviction
- component scores
- feature vector
- model/config version
- drivers and risks

This separation keeps reporting ergonomic while preserving the raw inputs needed for diagnostics and recalibration.

### 2. Enforce timestamp safety early

The workflow validates market data before signal generation and uses rolling or lagged transformations in the technical feature stack. Macro overlays only consume features observed at or before the signal timestamp.

The evaluation engine refuses to score unresolved horizons.

### 3. Make evaluation a first-class subsystem

Evaluation is not treated as a notebook-only backtest. It is a reusable module that:

- loads persisted signal snapshots
- aligns them to price history
- computes realized outcome metrics by horizon
- generates scorecards and degradation alerts
- persists detailed and summary artifacts

### 4. Govern adaptation

The adaptive engine only proposes parameter changes when:

- evaluated-signal sample size is large enough
- a holdout split exists
- candidate weights improve holdout hit rate and rank IC
- coefficient drift stays within configured bounds

Candidate versions are saved with evidence, linked back to their parent version, and promotion can remain manual.

### 5. Keep ingestion free-first

The provider layer now assumes:

- local CSV/parquet is the cleanest research input
- free public proxies are acceptable as optional fallbacks
- premium feeds remain optional integrations rather than hidden assumptions

This is especially important for India-first deployment, where contract fidelity often depends on user-curated local data.

## Current Flow

```text
price data
  -> market data validation
  -> feature construction
  -> regime / directional / inefficiency / macro overlays
  -> composite suggestion
  -> signal snapshot persistence
  -> forward outcome evaluation
  -> diagnostics + scorecards
  -> candidate parameter recalibration
  -> versioned recommendation or promotion
```

## Important Boundaries

- `data/ingestion/` owns provider abstraction and standardized loading.
- `signals/` owns inference-time logic.
- `analytics/evaluation/` owns realized-outcome measurement.
- `analytics/adaptation/` owns parameter governance and versioning.
- `analytics/backtest/` is now a compatibility facade over the evaluation/adaptation flow.
- `workflow/` coordinates the lifecycle but does not hide evaluation or adaptation internals.

## Near-Term Next Steps

- Add provider-backed active-contract and roll logic.
- Extend feature storage and experiment tracking metadata.
- Add richer event-study logic tied to official release calendars.
- Add approval tooling around candidate promotion.
