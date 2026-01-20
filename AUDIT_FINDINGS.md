# Hydra V3 Pipeline Audit Findings
**Date**: January 2025  
**Scope**: Full codebase audit for performance, bugs, and long-run stability

---

## 🔴 CRITICAL BUGS

### 1. Signal Engine Always Returns None
**File**: `src/stage3_v3/engine.py:411-412`  
**Impact**: **ALL SIGNALS BLOCKED** when `bias_hard_gate=True` (default)

```python
# CURRENT (BROKEN):
if bias_blocks:
    self.total_bias_blocked += 1
    logger.info(...)
return None  # ← WRONG: Outside the if block!

# FIX:
if bias_blocks:
    self.total_bias_blocked += 1
    logger.info(...)
    return None  # ← Move inside the if block
```

**Severity**: 🔴 Critical - No trades can fire with default config

---

### 2. API Key Hardcoded
**File**: `src/collectors/bootstrap.py:30`
```python
COINALYZE_API_KEY = "d02ff8e4-16e7-44b1-bcb8-ef663a8de294"
```
**Fix**: Move to environment variable or settings

---

## 🟡 PERFORMANCE BOTTLENECKS

### Stage 1 - Data Ingestion

| Issue | File | Line | Impact |
|-------|------|------|--------|
| `_update_buckets_loop` iterates ALL liquidations every 1s | `collectors/derivatives.py` | 656-748 | O(n) per symbol |
| numpy import inside methods | `collectors/bootstrap.py` | 80, 246-250 | Repeated import overhead |
| No rate limiting coordination between REST collectors | Multiple | - | Potential rate limit hits |

### Stage 2 - Feature Computation

| Issue | File | Line | Impact |
|-------|------|------|--------|
| `_build_profile_5m/30m` iterates ALL trades every 1s | `processors/structure.py` | 219-243 | O(n) per symbol |
| `_check_acceptance` iterates trades in 60s window | `processors/structure.py` | 394-441 | O(n) per call |
| `_compute_window_stats` iterates all events per window | `processors/liquidations.py` | 89-117 | O(n×3) per update |
| Untracked async task creation | `stage2/orchestrator.py` | 330 | Potential task leak |

### Stage 3 - Signal Engine

| Issue | File | Line | Impact |
|-------|------|------|--------|
| Creates new `MarketState` on every evaluate call | `engine.py` | 342-376 | Object allocation overhead |
| Synchronous signal evaluation for all signals | `engine.py` | 383-389 | Sequential processing |

---

## 🟢 ARCHITECTURE STRENGTHS

### Well-Designed Patterns
1. **Decoupled message queues** - WS receive → queue → worker pattern across all collectors
2. **Circuit breakers** with adaptive backoff in `src/core/resilience.py`
3. **12-hour periodic connection refresh** for long-run stability
4. **Singleton RegimeClassifier** with persistence
5. **Health monitoring** with auto-recovery callbacks
6. **Incremental EMA updates** in `TrendAnalyzer`

### Resilience Features
- `ConnectionSupervisor` with global pause capability
- Per-symbol stale thresholds in health monitor
- Graceful shutdown with task cancellation

---

## 📊 MEMORY MANAGEMENT

### Current Safeguards
- All buffers use `deque(maxlen=N)` - bounded memory
- `ResourceManager` class exists but underutilized
- Periodic health logging includes basic cleanup

### Potential Issues
| Buffer | Location | Max Size | Risk |
|--------|----------|----------|------|
| `_trades_5m` | `structure.py` | Unbounded deque | ⚠️ Memory growth |
| `_trades_30m` | `structure.py` | Unbounded deque | ⚠️ Memory growth |
| `_liquidations` | `derivatives.py` | 10,000 | ✅ Bounded |
| Signal `_last_signal` dicts | Various signals | Unbounded | ⚠️ Trimmed hourly |

---

## 🔧 RECOMMENDED OPTIMIZATIONS

### Priority 1: Fix Critical Bug
```python
# src/stage3_v3/engine.py line 411-412
if bias_blocks:
    self.total_bias_blocked += 1
    logger.info(...)
    return None  # Indent this line
```

### Priority 2: O(n) Loop Optimizations

**Option A: Incremental bucket updates**
```python
# Instead of rebuilding from scratch, maintain running totals
# and only subtract expired entries
```

**Option B: Time-bucketed data structures**
```python
# Use time-indexed buckets (e.g., per-minute buckets)
# Sum recent buckets instead of iterating all events
```

### Priority 3: Move API key to config
```python
# config/settings.py
COINALYZE_API_KEY = os.environ.get("COINALYZE_API_KEY", "")
```

### Priority 4: Bound trade buffers
```python
# structure.py - add maxlen
self._trades_5m: deque[Tuple[float, float, int]] = deque(maxlen=50000)
self._trades_30m: deque[Tuple[float, float, int]] = deque(maxlen=200000)
```

---

## 📈 PERFORMANCE IMPACT ESTIMATES

| Optimization | Expected Improvement |
|--------------|---------------------|
| Fix critical bug | **Enables trading** |
| Incremental bucket updates | ~50% CPU reduction in derivatives collector |
| Bounded trade buffers | Prevents unbounded memory growth |
| Move numpy imports to top | Minor (~1ms per call saved) |

---

## 🧪 TESTING RECOMMENDATIONS

1. **Unit test for engine.py bias gate** - Verify signals pass when bias allows
2. **Memory profiling** - Run for 24h+ and monitor memory growth
3. **Load test structure processor** - Measure latency with high trade volume
4. **Integration test** - Full pipeline with synthetic data

---

## 📁 FILES AUDITED

### Stage 1 - Data Ingestion
- `src/collectors/trades.py` ✅
- `src/collectors/orderbook.py` ✅
- `src/collectors/derivatives.py` ✅
- `src/collectors/bootstrap.py` ✅
- `src/core/models.py` ✅
- `src/core/resilience.py` ✅
- `src/stage1.py` ✅

### Stage 2 - Feature Computation
- `src/stage2/orchestrator.py` ✅
- `src/stage2/models.py` ✅
- `src/stage2/processors/order_flow.py` ✅
- `src/stage2/processors/structure.py` ✅
- `src/stage2/processors/liquidations.py` ✅
- `src/stage2/processors/absorption.py` ✅
- `src/stage2/processors/funding_oi.py` ✅
- `src/stage2/processors/alpha_state.py` ✅

### Stage 3 - Signal Engine
- `src/stage3_v3/engine.py` ✅
- `src/stage3_v3/models.py` ✅
- `src/stage3_v3/signals.py` ✅
- `src/stage3_v3/regime.py` ✅
- `src/stage3_v3/bias.py` ✅
- `src/stage3_v3/trend.py` ✅

### Pipeline & Dashboard
- `run_hydra.py` ✅
- `src/dashboard/global_runner_v3.py` ✅
- `src/dashboard/global_dashboard_v3.py` ✅
- `src/health/monitor.py` ✅

---

## ✅ NEXT STEPS

1. [ ] **IMMEDIATE**: Fix engine.py bias gate bug
2. [ ] Move API key to environment variable
3. [ ] Add maxlen to structure.py trade buffers
4. [ ] Implement incremental bucket updates for derivatives collector
5. [ ] Add unit tests for critical paths
6. [ ] Run 24h memory profiling test
