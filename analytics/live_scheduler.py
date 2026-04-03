"""
LiveFactorScheduler — Daemon that refreshes factor timing metrics on schedule.

Periodically:
1. Runs historical backtester on recent signals
2. Accumulates factor Sharpe ratios by regime
3. Updates FactorTimingEngine with latest metrics
4. Enables live signal generation to use current factor weights

Can run as background daemon or be triggered manually.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional

try:
    from apscheduler.schedulers.background import BackgroundScheduler
    HAS_APSCHEDULER = True
except ImportError:
    HAS_APSCHEDULER = False

from ..analytics.backtester import HistoricalBacktester
from ..analytics.factor_timing import FactorTimingEngine
from ..config.settings import settings
from ..data.storage.local import LocalStorage

logger = logging.getLogger(__name__)


class LiveFactorScheduler:
    """
    Schedules periodic factor metric refreshes.

    Can operate in two modes:
    1. Automatic (APScheduler): Runs in background via cron schedule
    2. Manual: Users call refresh() explicitly
    """

    def __init__(self, storage: Optional[LocalStorage] = None):
        self.storage = storage or LocalStorage()
        self.backtester = HistoricalBacktester(storage=storage)
        self.factor_timing = FactorTimingEngine(storage=storage)
        self.scheduler: Optional[BackgroundScheduler] = None
        self.is_running = False
        self.last_refresh_timestamp: Optional[datetime] = None
        self.last_refresh_commodities: List[str] = []

    def start_automatic_scheduler(
        self,
        commodities: List[str],
        refresh_hour: int = 0,  # UTC hour (default: midnight UTC)
        refresh_minute: int = 30,
    ) -> None:
        """
        Start background scheduler for daily factor refresh.

        Args:
            commodities: List of commodities to refresh
            refresh_hour: Hour of day (UTC) to run refresh
            refresh_minute: Minute of hour to run refresh
        """
        if not HAS_APSCHEDULER:
            logger.error(
                "APScheduler not installed. Install with: pip install apscheduler"
            )
            return

        if self.is_running:
            logger.warning("Scheduler already running")
            return

        self.scheduler = BackgroundScheduler()
        self.scheduler.add_job(
            self.refresh_all_commodities,
            "cron",
            hour=refresh_hour,
            minute=refresh_minute,
            args=[commodities],
            id="factor_refresh_daily",
        )

        self.scheduler.start()
        self.is_running = True
        logger.info(
            f"Factor refresh scheduler started (daily at {refresh_hour:02d}:{refresh_minute:02d} UTC)"
        )

    def stop_automatic_scheduler(self) -> None:
        """Stop background scheduler."""
        if self.scheduler and self.is_running:
            self.scheduler.shutdown()
            self.is_running = False
            logger.info("Factor refresh scheduler stopped")

    def refresh_all_commodities(self, commodities: List[str]) -> Dict[str, Dict]:
        """
        Manually trigger factor refresh for multiple commodities.

        Typical usage:
        - Called by automatic scheduler daily
        - Can be called manually for immediate refresh

        Args:
            commodities: List of commodity symbols

        Returns:
            Dict[commodity, refresh_diagnostics]
        """
        logger.info(f"Starting factor refresh for {len(commodities)} commodities...")
        results = {}

        for commodity in commodities:
            try:
                diag = self.refresh_commodity(commodity)
                results[commodity] = diag
            except Exception as e:
                logger.error(f"Failed to refresh {commodity}: {e}")
                results[commodity] = {"status": "error", "error": str(e)}

        self.last_refresh_timestamp = datetime.now()
        self.last_refresh_commodities = commodities

        logger.info(
            f"Factor refresh completed at {self.last_refresh_timestamp.isoformat()}"
        )
        return results

    def refresh_commodity(self, commodity: str) -> Dict:
        """
        Refresh factor metrics for a single commodity.

        1. Loads historical signal snapshots (last 90 days)
        2. Backtests against recent price data
        3. Accumulates regime-specific factor Sharpe ratios
        4. Persists metrics to storage

        Returns:
            Diagnostic info: {status, regimes_updated, factor_metrics, ...}
        """
        logger.info(f"Refreshing factor metrics for {commodity}...")

        try:
            # Load recent price data (last 90 days)
            price_data = self._load_recent_price_data(commodity, lookback_days=90)
            if price_data is None or price_data.empty:
                logger.warning(f"No recent price data for {commodity}")
                return {"status": "no_data", "commodity": commodity}

            # Run backtester
            evaluations_by_regime = self.backtester.backtest_commodity_historical(
                commodity=commodity,
                price_data=price_data,
                horizons=[1, 3, 5, 10, 20],
            )

            if not evaluations_by_regime:
                logger.warning(f"No backtest results for {commodity}")
                return {"status": "no_results", "commodity": commodity}

            # Generate diagnostic report
            report = self.backtester.generate_backtest_report(
                evaluations_by_regime, commodity
            )

            # Query updated factor metrics
            regimes_updated = list(evaluations_by_regime.keys())
            factor_diagnostics = {
                regime: self.factor_timing.get_factor_diagnostics(commodity, regime)
                for regime in regimes_updated
            }

            # Persist metrics
            self.factor_timing.persist_metrics(commodity)

            logger.info(
                f"Refreshed {commodity}: {len(regimes_updated)} regimes, "
                f"{sum(len(e) for e in evaluations_by_regime.values())} signals"
            )

            return {
                "status": "success",
                "commodity": commodity,
                "regimes_updated": regimes_updated,
                "n_signals": sum(len(e) for e in evaluations_by_regime.values()),
                "factor_diagnostics": factor_diagnostics,
                "report": report.to_dict("records") if not report.empty else [],
            }

        except Exception as e:
            logger.error(f"Error refreshing {commodity}: {e}")
            return {"status": "error", "commodity": commodity, "error": str(e)}

    def _load_recent_price_data(
        self, commodity: str, lookback_days: int = 90
    ) -> Optional[pd.DataFrame]:
        """
        Load recent price data for backtesting via MarketDataService.

        Falls back to None when no live or cached data is available.
        """
        try:
            from ..data.ingestion.market_data_service import MarketDataService
            svc = MarketDataService(storage=self.storage)
            end_date = datetime.now().date()
            start_date = end_date - timedelta(days=lookback_days)
            return svc.load_or_fetch_price_frame(
                commodity=commodity,
                start_date=start_date,
                end_date=end_date,
                refresh=False,
            )
        except Exception as e:
            logger.debug(f"Could not load price data for {commodity}: {e}")
            return None

    def get_refresh_status(self) -> Dict:
        """Get current scheduler status and last refresh info."""
        return {
            "is_running": self.is_running,
            "last_refresh": (
                self.last_refresh_timestamp.isoformat()
                if self.last_refresh_timestamp
                else None
            ),
            "last_refresh_commodities": self.last_refresh_commodities,
            "scheduler_type": "automatic" if HAS_APSCHEDULER else "manual",
        }


# Singleton instance
live_factor_scheduler = LiveFactorScheduler()
