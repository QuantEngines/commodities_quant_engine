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
- adaptive parameter recommendations with holdout validation and versioning
- explicit workflow orchestration for signal, evaluation, and adaptation cycles

## Architecture

### Core modules

- `config/`: typed settings for commodities, signal parameters, evaluation, adaptation, and storage
- `data/models.py`: canonical domain models for suggestions, signal snapshots, evaluations, and parameter versions
- `data/ingestion/`: local-first/free provider adapters, provider registry, and market-data service
- `data/quality_checks/`: market-data validation and stale/incomplete data detection
- `data/storage/`: local parquet/JSON artifact storage
- `regimes/`: regime classification
- `signals/`: directional, inefficiency, macro overlays, and composite suggestion engine
- `analytics/evaluation/`: realized-outcome evaluation engine
- `analytics/adaptation/`: governed parameter recalibration engine
- `workflow/`: end-to-end research lifecycle orchestration

### Lifecycle

1. Validate market data.
2. Build timestamp-safe technical features.
3. Generate regime, directional, inefficiency, and macro-adjusted composite signals.
4. Persist a `SignalSnapshot` artifact.
5. Evaluate historical snapshots against realized outcomes over 1D/3D/5D/10D/20D horizons.
6. Produce scorecards, confidence calibration, and degradation alerts.
7. Fit candidate parameter updates on historical evaluated signals using holdout validation.
8. Save candidate and active parameter versions with evidence and safety checks.

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

2. Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

3. Install dependencies:

```bash
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt
```

4. Run tests:

```bash
pytest -q
```

5. If importing from outside the repo root, export the parent directory to `PYTHONPATH`:

```bash
export PYTHONPATH="$(dirname "$PWD"):$PYTHONPATH"
```

### Windows Setup

1. Confirm Python 3.11+ is available:

```powershell
py --version
```

2. Create and activate a virtual environment:

```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
```

If PowerShell blocks activation, run:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

3. Install dependencies:

```powershell
py -m pip install --upgrade pip
py -m pip install -r requirements.txt
```

4. Run tests:

```powershell
py -m pytest -q
```

5. If importing from outside the repo root, set `PYTHONPATH` to the parent directory:

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

More schema detail is in [`docs/local_data_schema.md`](/Users/pramitdutta/Desktop/Trading%20Engines/commodities_quant_engine/docs/local_data_schema.md).

An example mapping file is available at [`config/local_data_example.yaml`](/Users/pramitdutta/Desktop/Trading%20Engines/commodities_quant_engine/config/local_data_example.yaml).

For curated local contract metadata, an example catalog is available at [`config/contract_master_example.csv`](/Users/pramitdutta/Desktop/Trading%20Engines/commodities_quant_engine/config/contract_master_example.csv). If you point `settings.contract_master.contract_catalog_path` to a real local catalog, the engine will prefer those contracts over deterministic fallback symbols when selecting the active contract.

For the optional `commodities-api.com` adapter, add your API key and explicit symbol map through settings or environment-backed overrides. It is treated as reference-data support, not as a replacement for native MCX futures history.

## Usage

### One-command local launcher

From the parent directory of this repo:

```bash
cd "/Users/pramitdutta/Desktop/Trading Engines"
python commodities_quant_engine/run_local.py
```

If you run it with no commodity flags in an interactive terminal, it now prompts you to choose:

- `All featured commodities` to combine only the featured top-traded contracts together
- featured high-liquidity contracts first: `GOLD`, `SILVER`, `CRUDEOIL`, `NATURALGAS`, `COPPER`, `ZINC`, and `LEAD`
- `Others` if you want to drill into the rest of the engine's broader MCX-aligned commodity universe

After an interactive selection, the launcher now enters live polling mode by default and refreshes every 30 seconds. It writes compact compressed live observations into `artifacts/live_signals/` and only persists full signal snapshots when the observed signal state materially changes, which keeps storage growth controlled.

To limit a live run during testing:

```bash
cd "/Users/pramitdutta/Desktop/Trading Engines"
python commodities_quant_engine/run_local.py --watch --refresh-seconds 30 --max-iterations 3 --commodity GOLD
```

To run with a real local file:

```bash
cd "/Users/pramitdutta/Desktop/Trading Engines"
python commodities_quant_engine/run_local.py --commodity GOLD --price-file commodities_quant_engine/local_data/market/gold_ohlcv.csv
```

To print the markdown suggestion as well:

```bash
cd "/Users/pramitdutta/Desktop/Trading Engines"
python commodities_quant_engine/run_local.py --show-markdown
```

To list configured commodities and their latest signal summaries:

```bash
cd "/Users/pramitdutta/Desktop/Trading Engines"
python commodities_quant_engine/run_local.py --list-commodities
```

To run all configured commodities in one go:

```bash
cd "/Users/pramitdutta/Desktop/Trading Engines"
python commodities_quant_engine/run_local.py --all-commodities
```

To run only selected commodities:

```bash
cd "/Users/pramitdutta/Desktop/Trading Engines"
python commodities_quant_engine/run_local.py --commodities GOLD SILVER COPPER
```

To use configured local/provider data for all commodities and skip missing ones cleanly:

```bash
cd "/Users/pramitdutta/Desktop/Trading Engines"
python commodities_quant_engine/run_local.py --all-commodities --use-provider --start-date 2025-01-01 --end-date 2025-12-31
```

To use the optional `commodities-api.com` provider for mapped commodities:

```bash
cd "/Users/pramitdutta/Desktop/Trading Engines"
python commodities_quant_engine/run_local.py --commodities GOLD SILVER --provider COMMODITIES_API --start-date 2025-01-01 --end-date 2025-12-31
```

The `COMMODITIES_API` path is intentionally opt-in. It is useful for reference/spot-style daily series where you have supplied a symbol map, but it should not be confused with exact MCX contract-chain data.

Live-mode observations are stored efficiently as compressed JSONL files, rotated by commodity and day. This is intentionally different from the slower research artifact path used for evaluation summaries and reports, because a 30-second polling loop would otherwise spend too much time rewriting parquet files.

### Execution assumptions

Signal evaluation and the compatibility backtest now use explicit execution assumptions rather than optimistic same-bar fills:

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

[`analytics/backtest`](/Users/pramitdutta/Desktop/Trading%20Engines/commodities_quant_engine/analytics/backtest/__init__.py) now acts as a compatibility layer over the newer workflow:

- `MacroBacktester` uses real price inputs and evaluates signals through `analytics/evaluation`
- `MacroParameterTuner` delegates candidate generation to the governed adaptive engine

There is now one intended research loop rather than a disconnected backtest path and a separate evaluation path.

## Configuration

Primary settings live in [`config/settings.py`](/Users/pramitdutta/Desktop/Trading%20Engines/commodities_quant_engine/config/settings.py).

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

- The engine remains a suggestion engine, not an execution engine.
- India-first assumptions are preserved in commodity defaults, contract metadata, and INR-oriented context.
- Free/local-first adapters still rely on user-supplied local files for best India-specific fidelity; public proxy tickers are fallbacks, not contract-perfect exchange replacements.
- Paid providers remain optional and disabled by default.
- No neural models were added; the adaptive layer uses holdout-validated statistical recalibration to stay explainable and robust.

## Extension Points

- Add exchange-specific contract metadata and richer roll logic on top of the local-first provider layer.
- Add richer curve/carry features once continuous-contract inputs are available.
- Extend evaluation with event-study diagnostics tied to official scheduled releases.
- Add manual approval tooling around candidate parameter promotion.
