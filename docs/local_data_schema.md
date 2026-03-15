# Local Data Schema

This project is local-first for research quality. The cleanest workflow is to maintain curated CSV/parquet files and point provider configs to them.

## Market Data

Required columns:

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

Notes:

- `timestamp` should be parseable by pandas.
- Files may be CSV or parquet.
- Intraday or daily bars both work, but all bars in one file should share a consistent frequency.

## Spot Data

Minimum columns:

- `timestamp`
- `close`

Recommended columns:

- `open`
- `high`
- `low`
- `volume`

## Macro Series

Required columns:

- `timestamp`
- `value`

Optional columns:

- `is_revised`
- `original_timestamp`

Examples:

- India CPI
- RBI policy rate
- USDINR
- PPAC domestic price references

## Macro Event Calendar

Required columns:

- `timestamp`
- `event_type`
- `title`

Optional columns:

- `expected_impact`
- `description`
- `source`
- `event_id`

Typical `event_type` values:

- `inflation_release`
- `central_bank_meeting`
- `gdp_release`
- `industrial_production`

## News Files

Required columns:

- `timestamp`
- `headline`

Optional columns:

- `content`
- `source`
- `url`
- `sentiment_score`
- `news_id`

These files are especially useful when you want to supplement free RSS ingestion with curated local headlines relevant to India commodities.

## Provider Mapping Guidance

The local-first provider layer typically maps keys such as:

- `GOLD`
- `CRUDEOIL`
- `spot:GOLD`
- `FX_USD_INR`
- `IN_CPI_YOY`

to file paths in the provider config.

## Recommendation

For production-minded local research:

- keep raw exchange downloads unchanged in a raw folder
- create cleaned standardized files for the engine
- version those cleaned files by date or batch
- document source provenance for each file
