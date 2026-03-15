# Macro Extension Implementation Plan

## Phase M1: Foundation & Interfaces
### 1.1 Architecture & Configuration
- [x] Create macro architecture note
- [x] Update folder structure with macro modules
- [x] Add macro data models to `data/models.py`
- [x] Extend `config/settings.py` with macro configurations
- [x] Create `config/macro_sources.yaml` and `config/macro_features.yaml`

### 1.2 Data Ingestion Framework
- [ ] Implement `MacroDataSource` abstract base class
- [ ] Create `OfficialMacroAdapter` for public sources (MOSPI, RBI)
- [ ] Create `BloombergMacroAdapter` stub with documented interface
- [ ] Create `ReutersMacroAdapter` stub with documented interface
- [ ] Create `GenericNewsAdapter` for fallback news ingestion
- [ ] Implement local storage integration for macro data
- [ ] Add macro data schemas and validation

### 1.3 Skeleton Modules
- [ ] Create skeleton for economic series ingestion
- [ ] Create skeleton for news ingestion
- [ ] Create skeleton for event calendar ingestion

## Phase M2: Feature Engineering
### 2.1 Core Feature Modules
- [ ] Implement `InflationFeatures` (CPI, WPI, trends, surprises)
- [ ] Implement `GrowthFeatures` (GDP, IP, PMI, acceleration)
- [ ] Implement `RatesFeatures` (policy rates, yields, real rates)
- [ ] Implement `FXFeatures` (USDINR, DXY, volatility)
- [ ] Implement `EventFeatures` (central bank events, release shocks)
- [ ] Implement `NewsFeatures` (sentiment, narrative intensity)

### 2.2 Feature Infrastructure
- [ ] Create macro feature registry
- [ ] Implement timestamp-safe transformations
- [ ] Add feature metadata tracking
- [ ] Create feature normalization utilities

## Phase M3: Signal Integration
### 3.1 Macro-Aware Signals
- [ ] Extend `RegimeEngine` with macro overlay
- [ ] Extend `DirectionalAlphaEngine` with macro features
- [ ] Implement `MacroConfidenceOverlay` module

### 3.2 Composite Engine Updates
- [ ] Update `CompositeDecisionEngine` to include macro confidence
- [ ] Modify suggestion scoring logic
- [ ] Update `Suggestion` model with macro fields

## Phase M4: Explainability & Testing
### 4.1 Reporting Enhancements
- [ ] Update markdown generator for macro context
- [ ] Add macro explanation utilities
- [ ] Create example notebooks

### 4.2 Testing & Documentation
- [ ] Add unit tests for macro components
- [ ] Create mock datasets for testing
- [ ] Update README with macro extension docs
- [ ] Add sample configurations

## Key Technical Decisions
- **Data Models**: Extend existing dataclasses in `data/models.py`
- **Configuration**: Use Pydantic settings with YAML fallbacks
- **Storage**: Leverage existing `LocalStorage` parquet system
- **Interfaces**: Follow `DataSource` and `FeatureEngine` patterns
- **Testing**: Add to existing `tests/` structure
- **Documentation**: Update README.md and add inline docs

## Risk Mitigation
- **Provider Dependencies**: All premium adapters are optional stubs
- **Data Availability**: Graceful degradation when macro data missing
- **Look-ahead Bias**: Strict timestamp validation in features
- **Complexity**: Keep macro contributions as overlays, not core logic
- **Explainability**: All macro effects must be traceable in outputs