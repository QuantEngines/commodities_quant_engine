"""
Event Risk Feature Engineering

Computes features related to macroeconomic events and release calendars.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any
from datetime import datetime, date, timedelta

from .base import MacroFeatureEngine
from ...data.models import MacroSeries, MacroEvent, MacroFeature

class EventFeatures(MacroFeatureEngine):
    """Computes event risk and calendar-related macroeconomic features."""

    def compute(self, macro_data: Dict[str, List[MacroSeries]], **kwargs) -> List[MacroFeature]:
        """Compute event features from macro series and event data."""
        features = []

        # Event calendar data
        event_data = kwargs.get('event_data', [])
        if event_data:
            event_features = self._compute_event_risk_features(event_data)
            features.extend(event_features)

        # Days since major events
        if event_data:
            days_since_features = self._compute_days_since_events(event_data)
            features.extend(days_since_features)

        # Event clustering and concentration
        if event_data:
            clustering_features = self._compute_event_clustering(event_data)
            features.extend(clustering_features)

        return features

    def _compute_event_risk_features(self, events: List[MacroEvent]) -> List[MacroFeature]:
        """Compute event risk window features."""
        features = []

        # Sort events by date
        sorted_events = sorted(events, key=lambda x: x.timestamp)

        for event in sorted_events:
            # Create risk window around event
            risk_start = event.timestamp.date() - timedelta(days=3)
            risk_end = event.timestamp.date() + timedelta(days=1)

            # Generate daily features for risk window
            current_date = risk_start
            while current_date <= risk_end:
                days_to_event = (event.timestamp.date() - current_date).days

                # Risk score based on proximity and impact
                base_risk = 1.0
                if event.expected_impact == 'high':
                    base_risk = 2.0
                elif event.expected_impact == 'low':
                    base_risk = 0.5

                # Distance-based decay
                distance_decay = max(0, 1 - abs(days_to_event) / 3)
                risk_score = base_risk * distance_decay

                features.append(self.create_macro_feature(
                    name='event_risk_window',
                    timestamp=datetime.combine(current_date, datetime.min.time()),
                    value=risk_score,
                    source_series=[f"EVENT_{event.event_id}"],
                    transform='event_risk_flag',
                    frequency='daily',
                    metadata={
                        'event_id': event.event_id,
                        'event_type': event.event_type,
                        'country': event.country,
                        'expected_impact': event.expected_impact,
                        'days_to_event': days_to_event
                    }
                ))

                current_date += timedelta(days=1)

        return features

    def _compute_days_since_events(self, events: List[MacroEvent]) -> List[MacroFeature]:
        """Compute days since major macro events."""
        features = []

        # Group events by type
        event_types = {}
        for event in events:
            if event.event_type not in event_types:
                event_types[event.event_type] = []
            event_types[event.event_type].append(event)

        # For each event type, compute days since last event
        for event_type, type_events in event_types.items():
            sorted_events = sorted(type_events, key=lambda x: x.timestamp)

            # Generate daily features from last event to now
            if sorted_events:
                last_event = sorted_events[-1]
                start_date = last_event.timestamp.date()
                end_date = date.today() + timedelta(days=30)  # Look ahead

                current_date = start_date
                while current_date <= end_date:
                    days_since = (current_date - start_date).days

                    features.append(self.create_macro_feature(
                        name=f'days_since_{event_type}',
                        timestamp=datetime.combine(current_date, datetime.min.time()),
                        value=days_since,
                        source_series=[f"EVENT_{event_type}"],
                        transform='days_since_event',
                        frequency='daily',
                        metadata={
                            'event_type': event_type,
                            'last_event_date': start_date.isoformat(),
                            'last_event_title': last_event.title
                        }
                    ))

                    current_date += timedelta(days=1)

        return features

    def _compute_event_clustering(self, events: List[MacroEvent]) -> List[MacroFeature]:
        """Compute event clustering and concentration features."""
        features = []

        # Sort events chronologically
        sorted_events = sorted(events, key=lambda x: x.timestamp)

        # Compute event density (events per week)
        if len(sorted_events) >= 2:
            date_range = (sorted_events[-1].timestamp - sorted_events[0].timestamp).days
            if date_range > 0:
                weeks = date_range / 7
                events_per_week = len(sorted_events) / weeks

                # Generate weekly density features
                current_date = sorted_events[0].timestamp.date()
                end_date = sorted_events[-1].timestamp.date()

                while current_date <= end_date:
                    # Count events in this week
                    week_start = current_date
                    week_end = current_date + timedelta(days=7)

                    week_events = [e for e in sorted_events
                                 if week_start <= e.timestamp.date() < week_end]

                    # High-impact events in week
                    high_impact_count = sum(1 for e in week_events if e.expected_impact == 'high')

                    features.append(self.create_macro_feature(
                        name='event_density_weekly',
                        timestamp=datetime.combine(current_date, datetime.min.time()),
                        value=len(week_events),
                        source_series=['EVENT_CALENDAR'],
                        transform='weekly_event_count',
                        frequency='weekly',
                        metadata={
                            'high_impact_events': high_impact_count,
                            'total_events': len(week_events)
                        }
                    ))

                    current_date += timedelta(days=7)

        return features

    def compute_event_risk_score(self, target_date: date, events: List[MacroEvent],
                               risk_window_days: int = 3) -> float:
        """Compute event risk score for a specific date."""
        total_risk = 0.0

        for event in events:
            days_diff = (event.timestamp.date() - target_date).days

            # Check if within risk window
            if abs(days_diff) <= risk_window_days:
                # Base risk by impact
                base_risk = 1.0
                if event.expected_impact == 'high':
                    base_risk = 2.0
                elif event.expected_impact == 'low':
                    base_risk = 0.5

                # Distance-based decay
                distance_decay = max(0, 1 - abs(days_diff) / risk_window_days)
                event_risk = base_risk * distance_decay

                total_risk += event_risk

        return min(total_risk, 5.0)  # Cap at 5.0

    def get_upcoming_high_impact_events(self, events: List[MacroEvent],
                                      days_ahead: int = 30) -> List[MacroEvent]:
        """Get high-impact events in the near future."""
        today = date.today()
        future_date = today + timedelta(days=days_ahead)

        upcoming = []
        for event in events:
            if (event.timestamp.date() >= today and
                event.timestamp.date() <= future_date and
                event.expected_impact == 'high'):
                upcoming.append(event)

        return sorted(upcoming, key=lambda x: x.timestamp)

    def classify_event_environment(self, events: List[MacroEvent], target_date: date) -> str:
        """Classify the event environment around a date."""
        risk_score = self.compute_event_risk_score(target_date, events)

        if risk_score >= 3.0:
            return 'high_event_risk'
        elif risk_score >= 1.5:
            return 'moderate_event_risk'
        elif risk_score >= 0.5:
            return 'low_event_risk'
        else:
            return 'no_event_risk'