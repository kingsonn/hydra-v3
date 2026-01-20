"""
Positioning Anomaly Detector
============================

Detects when current positioning state is UNUSUAL vs historical distribution.

Uses Isolation Forest on positioning state space.
Unusual states may precede significant moves.

This is learnable because unusual IS unusual (unsupervised).
No direction labels needed.
"""
import numpy as np
from typing import Optional
from collections import deque

from src.stage3_v2.models import LongHorizonState


class PositioningAnomalyDetector:
    """
    Anomaly detection on positioning state space.
    
    Uses Isolation Forest (or similar) to detect unusual
    combinations of positioning features.
    """
    
    def __init__(self, history_size: int = 10000):
        # State history for baseline (rolling window)
        self._state_history: deque[np.ndarray] = deque(maxlen=history_size)
        
        # Model (lazy initialization)
        self._model = None
        self._fitted = False
        self._min_samples_to_fit = 500
        self._refit_interval = 1000
        self._samples_since_fit = 0
        
        # Feature statistics for normalization
        self._feature_mean: Optional[np.ndarray] = None
        self._feature_std: Optional[np.ndarray] = None
    
    def add_state(self, state: LongHorizonState):
        """Add state to history and optionally refit model"""
        features = self._extract_features(state)
        self._state_history.append(features)
        self._samples_since_fit += 1
        
        # Refit periodically
        if (len(self._state_history) >= self._min_samples_to_fit and 
            self._samples_since_fit >= self._refit_interval):
            self._fit_model()
    
    def score(self, state: LongHorizonState) -> float:
        """
        Get anomaly score for current state.
        
        Returns:
            0-1 score (higher = more anomalous/unusual)
        """
        # Add to history
        self.add_state(state)
        
        if not self._fitted:
            return 0.5  # Neutral until we have enough data
        
        features = self._extract_features(state)
        normalized = self._normalize(features)
        
        try:
            # Isolation Forest returns -1 for outliers, +1 for inliers
            raw_score = self._model.decision_function([normalized])[0]
            
            # Convert to 0-1 (higher = more anomalous)
            # decision_function returns negative for anomalies
            anomaly_score = 1 / (1 + np.exp(raw_score * 2))  # Sigmoid
            
            return float(anomaly_score)
        except Exception:
            return 0.5
    
    def _extract_features(self, state: LongHorizonState) -> np.ndarray:
        """Extract feature vector for anomaly detection"""
        return np.array([
            state.funding_z,
            state.oi_delta_24h,
            state.liq_imbalance_4h,
            state.vol_expansion_ratio,
            state.depth_imbalance,
            state.absorption_z,
            state.oi_entry_displacement_pct,
            state.cumulative_funding_24h,
        ])
    
    def _normalize(self, features: np.ndarray) -> np.ndarray:
        """Normalize features using running statistics"""
        if self._feature_mean is None:
            return features
        
        std = np.where(self._feature_std > 1e-6, self._feature_std, 1.0)
        return (features - self._feature_mean) / std
    
    def _fit_model(self):
        """Fit Isolation Forest on historical states"""
        try:
            from sklearn.ensemble import IsolationForest
            
            X = np.array(self._state_history)
            
            # Update statistics
            self._feature_mean = np.mean(X, axis=0)
            self._feature_std = np.std(X, axis=0)
            
            # Normalize
            X_normalized = (X - self._feature_mean) / np.where(
                self._feature_std > 1e-6, self._feature_std, 1.0
            )
            
            # Fit model
            self._model = IsolationForest(
                contamination=0.1,  # 10% of states are "unusual"
                random_state=42,
                n_estimators=100,
            )
            self._model.fit(X_normalized)
            
            self._fitted = True
            self._samples_since_fit = 0
            
        except ImportError:
            # sklearn not available, use simple statistical approach
            self._fitted = False
        except Exception:
            self._fitted = False
    
    def get_percentile_score(self, state: LongHorizonState) -> float:
        """
        Fallback: Simple percentile-based anomaly score.
        
        Use if sklearn not available.
        """
        if len(self._state_history) < 100:
            return 0.5
        
        features = self._extract_features(state)
        X = np.array(self._state_history)
        
        # For each feature, compute percentile
        percentiles = []
        for i, val in enumerate(features):
            pct = np.mean(X[:, i] <= val) * 100
            # Distance from median (50th percentile)
            distance = abs(pct - 50) / 50
            percentiles.append(distance)
        
        # Average distance from median
        avg_distance = np.mean(percentiles)
        
        return float(avg_distance)
