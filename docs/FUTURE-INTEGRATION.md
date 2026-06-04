# Future Integration: ternary-kalman

## Current State
Provides Kalman filtering adapted for ternary state spaces with fixed-point arithmetic (Q16.16 format): `KalmanFilter` with predict/update cycle, `KalmanState` tracking estimate and covariance, and fixed-point math operators. No floating point — designed for bare-metal environments. Tracks state estimates that combine predictions with noisy observations.

## Integration Opportunities

### With ternary-cell (State Tracking Across Ticks)
ternary-cell's predict→perceive cycle mirrors the Kalman predict→update cycle. In predict, the cell computes expected state; in perceive, it receives actual observation. ternary-kalman formalizes this: the cell's prediction is the Kalman state estimate, the perception is the observation, and the Kalman update optimally combines them (weighting by prediction confidence vs. observation reliability). The covariance tracks cell uncertainty — cells with high covariance haven't learned their dynamics yet.

### With ternary-signals (Kalman-in-Frequency-Domain)
ternary-signals' DFT operates in the frequency domain. ternary-kalman tracks state in the time domain. Together: track dominant frequency over time using a Kalman filter on the DFT output. The filter smooths noisy frequency estimates across ticks, providing a stable spectral fingerprint even when individual DFT computations are noisy (especially at low tick counts).

### With ternary-streaming (Streaming State Estimation)
ternary-streaming's `StreamWindow` buffers tick data. ternary-kalman processes the buffer as a time series: each new tick is a Kalman observation, and the filter maintains a running state estimate with uncertainty bounds. The `RunningStatistics` from streaming can be derived from the Kalman state (mean = estimate, variance = covariance). This unifies streaming statistics with optimal state estimation.

## Potential in Mature Systems
In room-as-codespace, each room's state evolves over time with noise. ternary-kalman tracks room state optimally: predict the room's next state based on its dynamics, observe the actual state, and update the estimate with uncertainty-weighted fusion. PLATO monitors Kalman covariance across all rooms — high covariance rooms are uncertain (newly created, changing rapidly) and need more observations; low covariance rooms are well-tracked and can be sampled less frequently. On ESP32, the fixed-point Kalman filter tracks sensor state in real-time with <1µs per update.

## Cross-Pollination Ideas
- **ternary-fuzzy**: Fuzzy Kalman — use fuzzy membership for non-Gaussian uncertainty. When the Kalman assumption breaks down (bimodal distributions), fuzzy membership provides a fallback.
- **ternary-memory**: Kalman state as long-term memory — the state estimate IS the distilled memory of all observations, with covariance as the confidence.
- **ternary-control**: Kalman filter feeds into PID control — the filtered state estimate is the process variable, with uncertainty bounds informing the deadband.

## Dependencies for Next Steps
- Define `RoomKalmanFilter` tracking room state across ticks with fixed-point arithmetic
- Add Kalman update to ternary-cell's perceive phase as an alternative to raw observation
- Implement frequency-domain Kalman filter combining with ternary-signals DFT
- Benchmark Kalman predict/update on ESP32 at tick-rate frequencies (1kHz target)
