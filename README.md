# ternary-kalman

Kalman filtering for ternary state spaces with fixed-point arithmetic.

## Why This Exists

Standard Kalman filters operate on continuous real-valued state vectors. But many real systems have inherently three-valued signals — sensors that report below/normal/above, actuators that drive backward/off/forward, or decision boundaries that collapse to negative/neutral/positive. Bridging continuous estimation with discrete ternary decisions usually requires ad-hoc thresholding after the fact.

**ternary-kalman** unifies both: it runs a full predict/update Kalman cycle on continuous internal state, then projects to the nearest ternary vector {-1, 0, +1} when you need a discrete decision. A built-in fixed-point arithmetic layer (Q16.16) makes it suitable for embedded targets without FPU support.

## Core Concepts

| Type | Meaning |
|---|---|
| `Ternary` | A single ternary value: `Neg` (-1), `Zero` (0), or `Pos` (+1) |
| `FixedPoint` | Q16.16 fixed-point number — deterministic arithmetic for no-FPU environments |
| `TernaryObservation` | A vector of ternary readings (e.g. from a multi-sensor array) |
| `CovarianceMatrix` | Dense n×n covariance with matrix arithmetic (add, mul, scale, transpose, trace) |
| `TernaryKalmanFilter` | Full Kalman filter: predict → update → ternary projection |
| `StateEstimator` | Lightweight state + covariance wrapper with projection |

## Quick Start

```toml
# Cargo.toml
[dependencies]
ternary-kalman = "0.1"
```

```rust
use ternary_kalman::{TernaryKalmanFilter, TernaryObservation, Ternary};

fn main() {
    // 3-dimensional filter
    let mut kf = TernaryKalmanFilter::new(3);

    // Feed ternary observations
    let obs = TernaryObservation::new(vec![Ternary::Pos, Ternary::Zero, Ternary::Neg]);
    kf.step(&obs);

    // Read continuous state
    println!("state: {:?}", kf.state);

    // Project to ternary decisions
    let ternary = kf.ternary_state();
    println!("ternary: {:?}", ternary);
}
```

## API Overview

### Ternary
- `from_i8(i8) → Option<Ternary>` — safe conversion
- `to_i8()` / `to_f64()` — extract numeric value

### FixedPoint (Q16.16)
- `from_f64(f64)` / `to_f64()` — convert between representations
- `add`, `sub`, `mul`, `div` — deterministic fixed-point arithmetic

### TernaryObservation
- `new(Vec<Ternary>)` — wrap a ternary vector
- `to_f64_vec()` — convert for matrix math

### CovarianceMatrix
- `identity(n)`, `zeros(n)`, `diagonal(n, val)` — constructors
- `mat_mul`, `add`, `sub`, `scale`, `transpose`, `trace` — linear algebra

### TernaryKalmanFilter
- `new(dim)` — create with sensible defaults
- `predict()` — propagate state + covariance forward
- `update(&TernaryObservation)` — correct with ternary measurement
- `step(&TernaryObservation)` — predict + update in one call
- `ternary_state() → Vec<Ternary>` — project to discrete decisions
- `smooth(states, covs)` — backward smoothing pass
- `rts_smooth(states, covs, transition)` — Rauch-Tung-Striebel smoother

### Sensor Fusion
- `fuse_observations(observations, weights)` — weighted multi-sensor fusion

## How It Works

The filter maintains a continuous state vector **x** and covariance **P** internally. Each `predict()` step propagates state through the state transition matrix **F** and inflates covariance by the process noise **Q**: x ← Fx, P ← FPF' + Q.

Each `update()` step receives a ternary observation (mapped to {-1, 0, +1} floats), computes the Kalman gain **K** = PH'(HPH' + R)⁻¹, corrects the state x ← x + Ky, and shrinks the covariance P ← (I - KH)P.

The ternary projection thresholds each state dimension at ±0.5: values above map to +1, below to -1, and the rest to 0. This creates a natural deadband that suppresses noise-induced dithering near zero.

For time-series smoothing, `rts_smooth()` runs a backward pass that averages future filtered states into past estimates, reducing lag in the final output.

## Use Cases

- **Multi-sensor environmental monitoring** — fuse temperature, humidity, and pressure sensors into ternary below/normal/above classifications with proper uncertainty tracking
- **Embedded motor control** — run Kalman estimation in fixed-point on a Cortex-M0, projecting to forward/stop/reverse commands
- **Financial signal processing** — track continuous price estimates while outputting discrete bearish/neutral/bullish signals with quantified confidence

## Ecosystem

Part of the **SuperInstance** ternary computing ecosystem:

- [`ternary`](https://crates.io/crates/ternary) — core trit types and balanced ternary arithmetic
- [`ternary-kalman`](https://crates.io/crates/ternary-kalman) — this crate
- [`ternary-sensor`](https://crates.io/crates/ternary-sensor) — sensor classification and fusion
- [`ternary-control`](https://crates.io/crates/ternary-control) — ternary PID and bang-bang controllers
- [`ternary-fuzzy`](https://crates.io/crates/ternary-fuzzy) — fuzzy logic with ternary membership
- [`ternary-circuit`](https://crates.io/crates/ternary-circuit) — ternary logic gates and circuits

## License

MIT
