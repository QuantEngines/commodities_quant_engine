# Commodities Quant Engine

Research-first, human-in-the-loop commodities signal engine for India-focused deployment. The upgraded codebase now supports three explicit loops:

1. Generate auditable trade suggestions.
2. Evaluate those signals against realized forward outcomes.
3. Propose governed parameter recalibrations without uncontrolled self-modification.

## What Changed

The repository was refactored from a mostly scaffolded signal stack into a research workflow with:

- timestamp-safe feature engineering and market-data validation
- auditable signal snapshots with model/config versions
- free/local-first market and news ingestion adapters
- a broader default MCX-aligned commodity universe instead of a four-symbol demo set
- first-class signal evaluation across configurable horizons
- degradation and calibration diagnostics
- confidence reliability calibration and regime probability calibration artifacts
- drift dashboard generation with trailing-vs-history diagnostics
- adaptive parameter recommendations with holdout validation and versioning
- explicit workflow orchestration for signal, evaluation, and adaptation cycles

## Architecture

### Core modules

- `config/`: typed settings for commodities, signal parameters, evaluation, adaptation, and storage
- `data/models.py`: canonical domain models for suggestions, signal snapshots, evaluations, and parameter versions
- `data/ingestion/`: local-first/free provider adapters, provider registry, and market-data service
- `data/ingestion/shipping/`: shipping adapter interfaces plus local/manual/public stubs
- `data/quality_checks/`: market-data validation and stale/incomplete data detection
- `data/storage/`: local parquet/JSON artifact storage
- `shipping/`: shipping geographies, processing, features, and overlay logic
- `regimes/`: regime classification
- `signals/`: directional, inefficiency, macro overlays, and composite suggestion engine
- `analytics/evaluation/`: realized-outcome evaluation engine
- `analytics/adaptation/`: governed parameter recalibration engine
- `workflow/`: end-to-end research lifecycle orchestration

### Lifecycle

1. Validate market data.
2. Build timestamp-safe technical features.
3. Generate regime, directional, inefficiency, and macro-adjusted composite signals.
4. Optionally add shipping-intelligence context from config-driven geographies and flow features.
5. Persist a `SignalSnapshot` artifact.
6. Evaluate historical snapshots against realized outcomes over 1D/3D/5D/10D/20D horizons.
7. Produce scorecards, confidence calibration, regime calibration, and drift alerts.
8. Persist calibration artifacts that are consumed by live signal generation.
9. Fit candidate parameter updates on historical evaluated signals using purged walk-forward validation.
10. Save candidate and active parameter versions with evidence and safety checks.

### Trader-Facing Decision Framework

The signal engine now uses a structured decision hierarchy instead of a single flat score-to-label mapping:

1. Regime assessment
2. Directional bias
3. Entry quality
4. Signal agreement/conflict
5. Risk overlay
6. Composite trade recommendation

This improves coherence between internal model components and the final recommendation presented to users.

### Composite Math Notes (Conservative Update)

The composite engine was reviewed and adjusted conservatively with two goals: keep stable behavior that already worked, and fix only proven pathologies.

Preserved behavior:

- directional signal generation, horizon weighting, and clipping remain unchanged
- macro/shipping context engines remain unchanged
- risk-penalty construction remains additive and bounded
- calibration hooks (`confidence_calibration`, `regime_calibration`) remain in place

Targeted fixes applied:

- removed score double-counting in final action classification by using composite score directly for action thresholding
- reduced inefficiency dominance by replacing a raw linear term with a bounded nonlinearity (`tanh`)
- recalibrated entry-quality penalties to avoid systematic over-triggering of `Very Poor`
- replaced fabricated top-regime alternatives with evidence-first mapping:
  - use calibration map probabilities when available
  - otherwise use selected regime probability plus residual `other_regimes` mass

Interpretation split:

- direction is driven by directional stack and context
- entry quality is driven by inefficiency, extension, and volatility
- tradeability confidence blends direction, entry quality, agreement, and data quality, then applies calibration
- risk acts as penalty/gating, not directional reversal

### Entry Quality

`Entry Quality` is now a first-class decision field describing whether the current price location is attractive for execution.

It is derived from:

- pricing inefficiency
- recent price extension versus trend baseline
- mean-reversion pressure
- volatility context

Possible values:

- `Excellent`
- `Good`
- `Fair`
- `Poor`
- `Very Poor`

Entry quality directly affects the recommendation layer (for example, it can convert a directional bias into a wait-style recommendation).

### Confidence Decomposition

The engine now emits separate confidence dimensions instead of a single ambiguous number:

- `directional_confidence`: confidence in price direction
- `tradeability_confidence`: confidence that setup is executable now
- `data_quality_confidence`: confidence derived from input-data quality

The legacy `confidence_score` is preserved for backward compatibility, but trader-facing outputs emphasize the decomposed confidences.

### Recommendation Labels

Recommendations now follow a richer trader-oriented label set:

- `Strong Long Candidate`
- `Long Bias`
- `Long Bias / Wait for Pullback`
- `Watchlist Long`
- `Neutral / No Edge`
- `Watchlist Short`
- `Short Bias / Wait for Rally`
- `Short Bias`
- `Strong Short Candidate`
- `Regime Conflict / Avoid`

### Regime Output Improvements

Regime output now includes a top regime-probability stack (top 2-3 entries), and the selected regime is always assigned a valid non-zero probability.

### Explainability Improvements

Each suggestion now includes:

- normalized component contribution labels (for example, `Strong Positive`, `Mild Negative`)
- `dominant_component`
- `override_reason` when execution is intentionally suppressed
- explicit `supportive_signals`, `contradictory_signals`, and `key_risks`
- narrative explanation aligned to the final recommendation

### Output Schema (Trader View)

Primary fields in the suggestion output now include:

- commodity
- timestamp
- regime label + top regime probabilities
- directional bias
- entry quality
- trade recommendation
- directional/tradeability/data-quality confidence
- component contributions
- dominant component + override reason
- supportive signals
- contradictory signals
- key risks
- explanation summary

### Calibration And Drift Artifacts

Evaluation now writes commodity-level calibration artifacts to the evaluation store:

- `<COMMODITY>_calibration.json`: confidence anchors, bucket reliability, regime probability map, drift dashboard metrics
- `<COMMODITY>_drift_dashboard.md`: human-readable drift status and alert summary
- `<COMMODITY>_confidence_calibration.png`: confidence reliability curve (ideal vs calibrated)
- `<COMMODITY>_regime_calibration.png`: empirical vs calibrated regime alignment probabilities

Drift thresholds are now commodity-family aware (e.g., bullion, base metals, energy, agri) and configured under `settings.evaluation.drift_thresholds_by_family`.

`ResearchWorkflow.run_signal_cycle` automatically loads `<COMMODITY>_calibration.json` when available and applies:

- confidence calibration to final suggestion confidence scores
- regime probability calibration to regime probabilities stored in suggestions and snapshots

### Default MCX Commodity Universe

The default `settings.commodities` registry now covers a broad MCX-aligned commodity set rather than only `GOLD`, `SILVER`, `COPPER`, and `CRUDEOIL`.

- Bullion: `GOLD`, `GOLDM`, `GOLDGUINEA`, `GOLDPETAL`, `GOLDTEN`, `SILVER`, `SILVERM`, `SILVERMIC`, `SILVER1000`
- Base metals: `ALUMINIUM`, `ALUMINI`, `BRASSPHY`, `COPPER`, `COPPERM`, `LEAD`, `LEADMINI`, `NICKEL`, `NICKELM`, `STEELREBAR`, `TIN`, `ZINC`, `ZINCMINI`
- Energy: `BRCRUDEOIL`, `CRUDEOIL`, `CRUDEOILM`, `ELECDMBL`, `NATURALGAS`, `NATGASMINI`
- Agri and softs: `ALMOND`, `CARDAMOM`, `CASTORSEED`, `CHANA`, `CORIANDER`, `COTTON`, `COTTONCNDY`, `COTTONOIL`, `CPO`, `GUARGUM`, `GUARSEED`, `KAPAS`, `KAPASKHALI`, `MAIZE`, `MENTHAOIL`, `PEPPER`, `POTATO`, `RAWJUTE`, `RBDPALMOLEIN`, `RUBBER`, `SOYABEAN`, `SUGARMDEL`, `SUGARMKOL`, `SUGARSKLP`, `WHEAT`

Contract metadata for the major contracts is filled in directly. Less-liquid variants still use conservative placeholder metadata until you plug in a curated exchange contract master or local catalog.

## Folder Structure

```text
config/
data/
  contract_master/
  quality_checks/
  storage/
features/
regimes/
signals/
analytics/
  evaluation/
  adaptation/
workflow/
tests/
docs/
```

## Setup

Install dependencies:

```bash
pip install -r requirements.txt
```

The repository uses a flat-root package layout. Import it from the parent directory of `commodities_quant_engine`, or add that parent directory to `PYTHONPATH`.

### macOS Setup

1. Confirm Python 3.11+ is available:

```bash
python3 --version
```

1. Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

1. Install dependencies:

```bash
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt
```

1. Run tests:

```bash
pytest -q
```

1. If importing from outside the repo root, export the parent directory to `PYTHONPATH`:

```bash
export PYTHONPATH="$(dirname "$PWD"):$PYTHONPATH"
```

### Windows Setup

1. Confirm Python 3.11+ is available:

```powershell
py --version
```

1. Create and activate a virtual environment:

```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
```

If PowerShell blocks activation, run:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

1. Install dependencies:

```powershell
py -m pip install --upgrade pip
py -m pip install -r requirements.txt
```

1. Run tests:

```powershell
py -m pytest -q
```

1. If importing from outside the repo root, set `PYTHONPATH` to the parent directory:

```powershell
$env:PYTHONPATH = "$(Split-Path -Parent (Get-Location));$env:PYTHONPATH"
```

### Running The Engine

After setup on either OS, typical local usage is:

```python
from commodities_quant_engine import ResearchWorkflow

workflow = ResearchWorkflow()
```

## Free-First Data Sources

The ingestion layer now defaults to free or local sources:

- local CSV/parquet files are the primary path for OHLCV, spot, macro, FX, and weather data
- public/free fallbacks can be enabled per provider, such as Yahoo Finance style proxy tickers
- free RSS and local CSV are the default news inputs
- paid providers like Bloomberg and Reuters remain optional and disabled by default

This keeps the system usable in a local research setup without premium subscriptions, while preserving extension points for paid feeds later.

## Shipping Intelligence Layer

The repository now includes an optional shipping-intelligence subsystem for logistics-sensitive commodities.

- Config-driven ports, anchorages, chokepoints, and corridors live in `config/shipping_geographies.yaml`
- Adapter interfaces for AIS, port calls, route events, weather, and satellite context live under `data/ingestion/shipping/`
- Processing, aggregation, and first-pass features live under `shipping/`
- Composite suggestions can accept `shipping_feature_vectors` and turn them into bounded context rather than mandatory core inputs

Initial feature families include:

- port congestion
- anchorage buildup
- route disruption
- chokepoint stress
- tanker flow momentum
- shipping data quality scoring

The design is explicitly free/public-data-first today and leaves clean adapter seams for premium maritime feeds later.

See `docs/shipping_intelligence_architecture.md` for the architecture note.

### Shipping Demo

```bash
python run_shipping_intelligence_demo.py
```

### Shipping Usage Example

```python
from commodities_quant_engine import ResearchWorkflow, ShippingFeaturePipeline

workflow = ResearchWorkflow()
shipping_vectors = ShippingFeaturePipeline().run(
    commodity="CRUDEOIL",
    vessel_positions=my_local_vessel_positions_df,
    route_events=my_manual_route_events_df,
)

package = workflow.run_signal_cycle(
    commodity="CRUDEOIL",
    price_data=price_df,
    shipping_feature_vectors=shipping_vectors,
)
```

If no shipping vectors are supplied, the existing engine behavior is preserved.

### Shipping Market Benchmarks

The shipping layer now integrates multi-family shipping market benchmarks as first-class signal inputs. These optional indexes provide real-time context on transport cost pressures and help identify divergences between shipping costs and commodity valuations.

Supported benchmark families:

- **Dry-bulk** (for base metals and agri commodities): Baltic Dry Index (BDI), Baltic Capesize (BCI), Baltic Panamax (BPI), Baltic Supramax (BSI), and estimated bulker vessel values
- **Tanker** (for crude oil): Baltic Dirty Tanker Index (BDTI), Baltic Clean Tanker Index (BCTI), and estimated tanker vessel values
- **LNG** (for natural gas): LNG carrier rates and estimated LNG carrier vessel values

Each benchmark family is commodity-gated (e.g., dry-bulk features only compute for base metals/agri; tanker features only for crude oil). Per-commodity features include:

- benchmark level (normalized, zero-mean)
- benchmark momentum (strength of recent trend)
- benchmark shock flags (where applicable)
- shipping-vs-benchmark divergence (spread between observed logistics stress and predicted by index)

Benchmarks are aggregated with commodity-specific weights to produce generic `shipping_market_benchmark_zscore` and `shipping_market_divergence` outputs, while family-specific fields (e.g., `bdi_benchmark_zscore`, `bdti_shipping_divergence`) are preserved for research traceability.

Shipping benchmark vectors are automatically persisted to the research artifact store:

- parquet history: `artifacts/shipping/<COMMODITY>_benchmark_vectors.parquet` with flattened benchmark + divergence columns
- JSONL stream: `artifacts/shipping/<COMMODITY>_benchmark_vectors.jsonl.gz` (compressed)
- per-signal summary: `artifacts/shipping/<COMMODITY>_<SIGNAL_ID>_benchmark_summary.json`

This enables research on benchmark divergence patterns, seasonality, and relative predictiveness across commodity families.

## Sample Local Data Layout

For best India-specific fidelity, keep exchange and macro data in local CSV or parquet files and point provider configs at those files.

Example layout:

```text
local_data/
  market/
    gold_ohlcv.csv
    crudeoil_ohlcv.parquet
    cotton_spot.csv
  macro/
    in_cpi.csv
    in_rbi_rate.csv
    usd_inr.csv
    india_macro_calendar.csv
  news/
    macro_news.csv
```

### OHLCV schema

Expected columns for market data files:

- `timestamp`
- `open`
- `high`
- `low`
- `close`
- `volume`

Optional columns:

- `open_interest`
- `spot_close`
- `next_close`

Example:

```csv
timestamp,open,high,low,close,volume,open_interest
2025-01-01 09:15:00,62500,62620,62440,62590,18420,125000
2025-01-02 09:15:00,62590,62780,62520,62710,20110,127450
```

### Macro series schema

Expected columns for macro series files:

- `timestamp`
- `value`

Optional columns:

- `is_revised`
- `original_timestamp`

Example:

```csv
timestamp,value
2025-01-31,5.10
2025-02-28,5.24
```

### Macro event calendar schema

Expected columns:

- `timestamp`
- `event_type`
- `title`

Optional columns:

- `expected_impact`
- `description`
- `source`

Example:

```csv
timestamp,event_type,title,expected_impact,source
2025-02-12 17:30:00,inflation_release,India CPI Release,high,MOSPI
2025-02-07 10:00:00,central_bank_meeting,RBI Policy Decision,high,RBI
```

### News CSV schema

Expected columns:

- `timestamp`
- `headline`

Optional columns:

- `content`
- `source`
- `url`
- `sentiment_score`

Example:

```csv
timestamp,headline,content,source
2025-02-10 08:00:00,Gold rises as real yields ease,Bullion gained in early trade,LOCAL
2025-02-10 09:30:00,Oil slips on demand concerns,Crude prices softened after weak demand data,LOCAL
```

More schema detail is in [`docs/local_data_schema.md`](docs/local_data_schema.md).

An example mapping file is available at [`config/local_data_example.yaml`](config/local_data_example.yaml).

For curated local contract metadata, an example catalog is available at [`config/contract_master_example.csv`](config/contract_master_example.csv). If you point `settings.contract_master.contract_catalog_path` to a real local catalog, the engine will prefer those contracts over deterministic fallback symbols when selecting the active contract.

For the optional `commodities-api.com` adapter, add your API key and explicit symbol map through settings or environment-backed overrides. It is treated as reference-data support, not as a replacement for native MCX futures history.

## Usage

### One-command local launcher

From the parent directory of this repo:

```bash
cd /path/to/parent/of/commodities_quant_engine
python commodities_quant_engine/main.py
```

If you run it with no commodity flags in an interactive terminal, it now prompts you to choose:

- `All featured commodities` to combine only the featured top-traded contracts together
- featured high-liquidity contracts first: `GOLD`, `SILVER`, `CRUDEOIL`, `NATURALGAS`, `COPPER`, `ZINC`, and `LEAD`
- `Others` if you want to drill into the rest of the engine's broader MCX-aligned commodity universe

After an interactive selection, the launcher now enters live polling mode by default and refreshes every 30 seconds. It writes compact compressed live observations into `artifacts/live_signals/` and only persists full signal snapshots when the observed signal state materially changes, which keeps storage growth controlled.

To limit a live run during testing:

```bash
cd /path/to/parent/of/commodities_quant_engine
python commodities_quant_engine/main.py --watch --refresh-seconds 30 --max-iterations 3 --commodity GOLD
```

To run with a real local file:

```bash
cd /path/to/parent/of/commodities_quant_engine
python commodities_quant_engine/main.py --commodity GOLD --price-file commodities_quant_engine/local_data/market/gold_ohlcv.csv
```

To print the markdown suggestion as well:

```bash
cd /path/to/parent/of/commodities_quant_engine
python commodities_quant_engine/main.py --show-markdown
```

To list configured commodities and their latest signal summaries:

```bash
cd /path/to/parent/of/commodities_quant_engine
python commodities_quant_engine/main.py --list-commodities
```

To run all configured commodities in one go:

```bash
cd /path/to/parent/of/commodities_quant_engine
python commodities_quant_engine/main.py --all-commodities
```

To run only selected commodities:

```bash
cd /path/to/parent/of/commodities_quant_engine
python commodities_quant_engine/main.py --commodities GOLD SILVER COPPER
```

To use configured local/provider data for all commodities and skip missing ones cleanly:

```bash
cd /path/to/parent/of/commodities_quant_engine
python commodities_quant_engine/main.py --all-commodities --use-provider --start-date 2025-01-01 --end-date 2025-12-31
```

To use the optional `commodities-api.com` provider for mapped commodities:

```bash
cd /path/to/parent/of/commodities_quant_engine
python commodities_quant_engine/main.py --commodities GOLD SILVER --provider COMMODITIES_API --start-date 2025-01-01 --end-date 2025-12-31
```

The `COMMODITIES_API` path is intentionally opt-in. It is useful for reference/spot-style daily series where you have supplied a symbol map, but it should not be confused with exact MCX contract-chain data.

Live-mode observations are stored efficiently as compressed JSONL files, rotated by commodity and day. This is intentionally different from the slower research artifact path used for evaluation summaries and reports, because a 30-second polling loop would otherwise spend too much time rewriting parquet files.

### Evaluation Pricing Assumptions

Signal evaluation and the compatibility backtest now use explicit evaluation-pricing assumptions rather than optimistic same-bar fills:

- signal issuance at bar close, with default entry on the next bar
- configurable `open`/`close` entry and exit price fields
- adverse slippage applied on entry and exit
- higher slippage when bar volume is thin relative to recent history
- contract catalog provenance retained so you can see when the engine is using fallback symbols versus curated contract metadata

### End-to-end workflow

```python
import pandas as pd
from commodities_quant_engine import ResearchWorkflow

price_data = pd.read_parquet("gold_prices.parquet").set_index("timestamp")
workflow = ResearchWorkflow()

signal_package = workflow.run_signal_cycle(
    commodity="GOLD",
    price_data=price_data.iloc[:-10],
)

evaluation = workflow.run_evaluation_cycle(
    commodity="GOLD",
    price_data=price_data,
)

decision = workflow.run_adaptation_cycle(
    commodity="GOLD",
    dry_run=True,
)
```

### Direct signal generation

```python
from commodities_quant_engine.signals.composite.composite_decision import CompositeDecisionEngine

engine = CompositeDecisionEngine()
package = engine.generate_signal_package(price_data, "GOLD")
print(package.suggestion.to_markdown())
```

### Standardized market-data loading

```python
from datetime import date
from commodities_quant_engine.data.ingestion import market_data_service

price_data = market_data_service.load_price_frame(
    commodity="GOLD",
    start_date=date(2024, 1, 1),
    end_date=date(2025, 1, 1),
)
```

## Stored Artifacts

Artifacts default to `artifacts/` and are split into:

- `signals/`: persisted `SignalSnapshot` history
- `evaluations/`: detailed realized-outcome records and summary scorecards
- `parameters/`: active, incumbent, and candidate parameter versions plus decision logs
- `reports/`: saved signal markdown payloads and audit traces

## Signal Evaluation

The evaluation engine compares persisted signals with realized market paths using trading-day horizons. It reports:

- hit rate and directional confusion
- average signed return and volatility-adjusted return
- maximum favorable and adverse excursion
- follow-through and reversal diagnostics
- rank IC and confidence calibration
- regime and confidence-bucket breakdowns
- degradation alerts for recent underperformance

Evaluation is timestamp-safe: unresolved forward windows are skipped, and signals are aligned to information available at issuance time.

## Adaptive Learning

The adaptive engine uses evaluated signals to recommend recalibration of directional feature weights. It is intentionally governed:

- minimum evaluated sample size is required
- candidate weights are fit on a training split and checked on a holdout split
- candidate must improve holdout hit rate and rank IC
- excessive parameter drift is rejected
- versions are persisted with parent linkage and evidence
- dry-run and recommendation-only mode are supported by default
- manual approval remains the default promotion posture

This keeps the system adaptive, but not self-modifying in an opaque way.

## Legacy Backtest Compatibility

[`analytics/backtest`](analytics/backtest/__init__.py) now acts as a compatibility layer over the newer workflow:

- `MacroBacktester` uses real price inputs and evaluates signals through `analytics/evaluation`
- `MacroParameterTuner` delegates candidate generation to the governed adaptive engine

There is now one intended research loop rather than a disconnected backtest path and a separate evaluation path.

## Configuration

Primary settings live in [`config/settings.py`](config/settings.py).

Important sections:

- `settings.signal`: feature names, default weights, horizons, staleness rules
- `settings.composite`: aggregation weights and suggestion thresholds
- `settings.evaluation`: horizons, confidence buckets, degradation window
- `settings.adaptation`: minimum sample size, holdout fraction, drift limits, promotion controls
- `settings.storage`: artifact locations

## Testing

Run:

```bash
pytest -q
```

Current tests cover:

- config and domain models
- free/local-first provider adapters
- backtest compatibility wrapper
- composite signal generation
- signal persistence workflow
- timestamp-safe evaluation
- adaptive update recommendation safeguards

## Assumptions And Limits

- The engine remains a suggestion engine, not an order-placement or broker-execution engine.
- India-first assumptions are preserved in commodity defaults, contract metadata, and INR-oriented context.
- Free/local-first adapters still rely on user-supplied local files for best India-specific fidelity; public proxy tickers are fallbacks, not contract-perfect exchange replacements.
- Paid providers remain optional and disabled by default.
- No neural models were added; the adaptive layer uses holdout-validated statistical recalibration to stay explainable and robust.

## Extension Points

- Add exchange-specific contract metadata and richer roll logic on top of the local-first provider layer.
- Add richer curve/carry features once continuous-contract inputs are available.
- Extend evaluation with event-study diagnostics tied to official scheduled releases.
- Add manual approval tooling around candidate parameter promotion.
