"""
Signal Context Similarity
=========================

Finds historical episodes SIMILAR to current state and extracts outcomes.

Uses embedding + nearest neighbor search to find similar market structures.
Returns historical win rates for similar contexts.

This is robust because we're finding similar structures, not predicting direction.
"""
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from collections import deque
import time

from src.stage3_v2.models import LongHorizonState, SignalType, Direction


@dataclass
class HistoricalEpisode:
    """Record of a historical signal episode"""
    timestamp: int
    symbol: str
    signal_type: str
    direction: str
    context_features: np.ndarray
    
    # Outcome
    outcome: str  # "win", "loss", "timeout"
    holding_hours: float
    pnl_pct: float
    max_adverse_pct: float  # Max drawdown during hold


class SignalContextSimilarity:
    """
    Find similar historical signal contexts and extract outcomes.
    
    Uses simple Euclidean distance on normalized feature vectors.
    Can be upgraded to use FAISS or learned embeddings.
    """
    
    def __init__(self, max_episodes: int = 5000):
        self.episodes: List[HistoricalEpisode] = []
        self.max_episodes = max_episodes
        
        # Feature statistics for normalization
        self._feature_mean: Optional[np.ndarray] = None
        self._feature_std: Optional[np.ndarray] = None
        
        # Minimum episodes needed for meaningful similarity
        self._min_episodes = 50
    
    def add_episode(self, episode: HistoricalEpisode):
        """Add a completed episode to history"""
        self.episodes.append(episode)
        
        # Trim if over limit (keep recent)
        if len(self.episodes) > self.max_episodes:
            self.episodes = self.episodes[-self.max_episodes:]
        
        # Update statistics periodically
        if len(self.episodes) % 100 == 0:
            self._update_statistics()
    
    def record_signal_entry(
        self,
        state: LongHorizonState,
        signal_type: SignalType,
        direction: Direction,
    ) -> int:
        """
        Record a new signal entry. Returns episode_id for later completion.
        """
        features = self._extract_features(state)
        
        episode = HistoricalEpisode(
            timestamp=state.timestamp_ms,
            symbol=state.symbol,
            signal_type=signal_type.value,
            direction=direction.value,
            context_features=features,
            outcome="pending",
            holding_hours=0.0,
            pnl_pct=0.0,
            max_adverse_pct=0.0,
        )
        
        self.episodes.append(episode)
        return len(self.episodes) - 1  # Return index as ID
    
    def complete_episode(
        self,
        episode_id: int,
        outcome: str,
        holding_hours: float,
        pnl_pct: float,
        max_adverse_pct: float,
    ):
        """Complete a pending episode with outcome"""
        if 0 <= episode_id < len(self.episodes):
            ep = self.episodes[episode_id]
            ep.outcome = outcome
            ep.holding_hours = holding_hours
            ep.pnl_pct = pnl_pct
            ep.max_adverse_pct = max_adverse_pct
    
    def find_similar(
        self,
        state: LongHorizonState,
        signal_type: Optional[SignalType] = None,
        k: int = 20,
    ) -> List[HistoricalEpisode]:
        """
        Find k most similar historical episodes.
        
        Args:
            state: Current market state
            signal_type: Filter to same signal type (optional)
            k: Number of similar episodes to return
        """
        if len(self.episodes) < self._min_episodes:
            return []
        
        current_features = self._extract_features(state)
        current_normalized = self._normalize(current_features)
        
        # Filter completed episodes
        candidates = [
            ep for ep in self.episodes 
            if ep.outcome != "pending"
            and (signal_type is None or ep.signal_type == signal_type.value)
        ]
        
        if not candidates:
            return []
        
        # Compute distances
        distances = []
        for ep in candidates:
            ep_normalized = self._normalize(ep.context_features)
            dist = np.linalg.norm(current_normalized - ep_normalized)
            distances.append((dist, ep))
        
        # Sort by distance and return top k
        distances.sort(key=lambda x: x[0])
        return [ep for _, ep in distances[:k]]
    
    def get_historical_win_rate(
        self,
        state: LongHorizonState,
        signal_type: Optional[SignalType] = None,
    ) -> float:
        """
        Get historical win rate for similar contexts.
        
        Returns:
            Win rate (0-1), defaults to 0.5 if insufficient data
        """
        similar = self.find_similar(state, signal_type, k=20)
        
        if len(similar) < 5:
            return 0.5  # Not enough data
        
        wins = sum(1 for ep in similar if ep.outcome == "win")
        return wins / len(similar)
    
    def get_context_statistics(
        self,
        state: LongHorizonState,
        signal_type: Optional[SignalType] = None,
    ) -> Dict:
        """
        Get detailed statistics for similar historical contexts.
        """
        similar = self.find_similar(state, signal_type, k=20)
        
        if len(similar) < 5:
            return {
                "win_rate": 0.5,
                "avg_holding_hours": 24,
                "avg_pnl_pct": 0.0,
                "avg_max_adverse_pct": 0.02,
                "sample_size": len(similar),
                "sufficient_data": False,
            }
        
        wins = sum(1 for ep in similar if ep.outcome == "win")
        
        return {
            "win_rate": wins / len(similar),
            "avg_holding_hours": np.mean([ep.holding_hours for ep in similar]),
            "avg_pnl_pct": np.mean([ep.pnl_pct for ep in similar]),
            "avg_max_adverse_pct": np.mean([ep.max_adverse_pct for ep in similar]),
            "sample_size": len(similar),
            "sufficient_data": True,
        }
    
    def _extract_features(self, state: LongHorizonState) -> np.ndarray:
        """Extract feature vector for similarity search"""
        return np.array([
            state.funding_z,
            state.funding_z_8h_avg,
            state.oi_delta_4h,
            state.oi_delta_24h,
            state.oi_entry_displacement_pct,
            state.liq_imbalance_4h,
            state.liq_total_usd_24h / 1_000_000,  # Normalize to millions
            state.vol_expansion_ratio,
            state.vol_compression_duration_h / 24,  # Normalize to days
            state.depth_imbalance,
            state.absorption_z,
        ])
    
    def _normalize(self, features: np.ndarray) -> np.ndarray:
        """Normalize features for distance computation"""
        if self._feature_mean is None:
            return features
        
        std = np.where(self._feature_std > 1e-6, self._feature_std, 1.0)
        return (features - self._feature_mean) / std
    
    def _update_statistics(self):
        """Update feature statistics for normalization"""
        if len(self.episodes) < 50:
            return
        
        features = np.array([ep.context_features for ep in self.episodes])
        self._feature_mean = np.mean(features, axis=0)
        self._feature_std = np.std(features, axis=0)
    
    def save(self, filepath: str):
        """Save episodes to file"""
        import pickle
        with open(filepath, 'wb') as f:
            pickle.dump({
                'episodes': self.episodes,
                'feature_mean': self._feature_mean,
                'feature_std': self._feature_std,
            }, f)
    
    def load(self, filepath: str):
        """Load episodes from file"""
        import pickle
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
            self.episodes = data['episodes']
            self._feature_mean = data.get('feature_mean')
            self._feature_std = data.get('feature_std')
