#![forbid(unsafe_code)]

//! Kalman filter adapted for ternary state spaces with fixed-point arithmetic.

/// Ternary value: -1, 0, or +1.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Ternary {
    Neg = -1,
    Zero = 0,
    Pos = 1,
}

impl Ternary {
    pub fn from_i8(v: i8) -> Option<Self> {
        match v {
            -1 => Some(Ternary::Neg),
            0 => Some(Ternary::Zero),
            1 => Some(Ternary::Pos),
            _ => None,
        }
    }

    pub fn to_i8(self) -> i8 {
        self as i8
    }

    pub fn to_f64(self) -> f64 {
        self as i8 as f64
    }
}

/// Fixed-point number using i32 with 16 fractional bits (Q16.16).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub struct FixedPoint(i32);

impl FixedPoint {
    pub const SCALE: i32 = 1 << 16;

    pub fn from_f64(v: f64) -> Self {
        FixedPoint((v * Self::SCALE as f64) as i32)
    }

    pub fn to_f64(self) -> f64 {
        self.0 as f64 / Self::SCALE as f64
    }

    pub fn from_i32(v: i32) -> Self {
        FixedPoint(v * Self::SCALE)
    }

    pub fn zero() -> Self {
        FixedPoint(0)
    }

    pub fn one() -> Self {
        FixedPoint(Self::SCALE)
    }

    pub fn add(self, other: Self) -> Self {
        FixedPoint(self.0 + other.0)
    }

    pub fn sub(self, other: Self) -> Self {
        FixedPoint(self.0 - other.0)
    }

    pub fn mul(self, other: Self) -> Self {
        FixedPoint((self.0 as i64 * other.0 as i64 >> 16) as i32)
    }

    pub fn div(self, other: Self) -> Self {
        FixedPoint(((self.0 as i64 * (1i64 << 16)) / other.0 as i64) as i32)
    }
}

/// Ternary observation: a vector of ternary values.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TernaryObservation {
    pub values: Vec<Ternary>,
}

impl TernaryObservation {
    pub fn new(values: Vec<Ternary>) -> Self {
        TernaryObservation { values }
    }

    pub fn len(&self) -> usize {
        self.values.len()
    }

    pub fn is_empty(&self) -> bool {
        self.values.is_empty()
    }

    pub fn to_f64_vec(&self) -> Vec<f64> {
        self.values.iter().map(|t| t.to_f64()).collect()
    }
}

/// Covariance matrix on a ternary grid, stored as a dense Vec<Vec<f64>>.
#[derive(Clone, Debug)]
pub struct CovarianceMatrix {
    pub n: usize,
    pub data: Vec<Vec<f64>>,
}

impl CovarianceMatrix {
    pub fn identity(n: usize) -> Self {
        let mut data = vec![vec![0.0; n]; n];
        for i in 0..n {
            data[i][i] = 1.0;
        }
        CovarianceMatrix { n, data }
    }

    pub fn zeros(n: usize) -> Self {
        CovarianceMatrix {
            n,
            data: vec![vec![0.0; n]; n],
        }
    }

    pub fn diagonal(n: usize, val: f64) -> Self {
        let mut m = Self::zeros(n);
        for i in 0..n {
            m.data[i][i] = val;
        }
        m
    }

    pub fn get(&self, i: usize, j: usize) -> f64 {
        self.data[i][j]
    }

    pub fn set(&mut self, i: usize, j: usize, val: f64) {
        self.data[i][j] = val;
    }

    /// Add another matrix (element-wise).
    pub fn add(&self, other: &Self) -> Self {
        let mut result = Self::zeros(self.n);
        for i in 0..self.n {
            for j in 0..self.n {
                result.data[i][j] = self.data[i][j] + other.data[i][j];
            }
        }
        result
    }

    /// Subtract another matrix (element-wise).
    pub fn sub(&self, other: &Self) -> Self {
        let mut result = Self::zeros(self.n);
        for i in 0..self.n {
            for j in 0..self.n {
                result.data[i][j] = self.data[i][j] - other.data[i][j];
            }
        }
        result
    }

    /// Matrix multiply self * other.
    pub fn mat_mul(&self, other: &Self) -> Self {
        let mut result = Self::zeros(self.n);
        for i in 0..self.n {
            for j in 0..self.n {
                let mut sum = 0.0;
                for k in 0..self.n {
                    sum += self.data[i][k] * other.data[k][j];
                }
                result.data[i][j] = sum;
            }
        }
        result
    }

    /// Scale all elements by a scalar.
    pub fn scale(&self, s: f64) -> Self {
        let mut result = Self::zeros(self.n);
        for i in 0..self.n {
            for j in 0..self.n {
                result.data[i][j] = self.data[i][j] * s;
            }
        }
        result
    }

    /// Transpose.
    pub fn transpose(&self) -> Self {
        let mut result = Self::zeros(self.n);
        for i in 0..self.n {
            for j in 0..self.n {
                result.data[i][j] = self.data[j][i];
            }
        }
        result
    }

    /// Trace (sum of diagonal).
    pub fn trace(&self) -> f64 {
        (0..self.n).map(|i| self.data[i][i]).sum()
    }
}

/// Simple matrix-vector multiply.
pub fn mat_vec_mul(mat: &CovarianceMatrix, vec: &[f64]) -> Vec<f64> {
    let n = mat.n;
    let mut result = vec![0.0; n];
    for i in 0..n {
        for j in 0..n {
            result[i] += mat.data[i][j] * vec[j];
        }
    }
    result
}

/// State estimator: maintains a state vector and covariance.
#[derive(Clone, Debug)]
pub struct StateEstimator {
    pub state: Vec<f64>,
    pub covariance: CovarianceMatrix,
}

impl StateEstimator {
    pub fn new(initial_state: Vec<f64>, initial_covariance: CovarianceMatrix) -> Self {
        StateEstimator {
            state: initial_state,
            covariance: initial_covariance,
        }
    }

    pub fn dim(&self) -> usize {
        self.state.len()
    }

    /// Project state to nearest ternary values.
    pub fn ternary_projection(&self) -> Vec<Ternary> {
        self.state
            .iter()
            .map(|&v| {
                if v > 0.5 {
                    Ternary::Pos
                } else if v < -0.5 {
                    Ternary::Neg
                } else {
                    Ternary::Zero
                }
            })
            .collect()
    }
}

/// Kalman filter adapted for ternary state spaces.
#[derive(Clone, Debug)]
pub struct TernaryKalmanFilter {
    pub dim: usize,
    pub state: Vec<f64>,
    pub covariance: CovarianceMatrix,
    pub process_noise: CovarianceMatrix,
    pub measurement_noise: CovarianceMatrix,
    pub state_transition: CovarianceMatrix,
    pub observation_matrix: CovarianceMatrix,
}

impl TernaryKalmanFilter {
    pub fn new(dim: usize) -> Self {
        TernaryKalmanFilter {
            dim,
            state: vec![0.0; dim],
            covariance: CovarianceMatrix::identity(dim),
            process_noise: CovarianceMatrix::diagonal(dim, 0.01),
            measurement_noise: CovarianceMatrix::diagonal(dim, 0.1),
            state_transition: CovarianceMatrix::identity(dim),
            observation_matrix: CovarianceMatrix::identity(dim),
        }
    }

    /// Predict step: propagate state and covariance forward.
    pub fn predict(&mut self) {
        // state = F * state
        self.state = mat_vec_mul(&self.state_transition, &self.state);

        // P = F * P * F' + Q
        let fp = self.state_transition.mat_mul(&self.covariance);
        let fpft = fp.mat_mul(&self.state_transition.transpose());
        self.covariance = fpft.add(&self.process_noise);
    }

    /// Update step with a ternary observation.
    pub fn update(&mut self, observation: &TernaryObservation) {
        let z: Vec<f64> = observation.to_f64_vec();

        // Innovation: y = z - H * x
        let hx = mat_vec_mul(&self.observation_matrix, &self.state);
        let y: Vec<f64> = z.iter().zip(hx.iter()).map(|(zi, hxi)| zi - hxi).collect();

        // S = H * P * H' + R
        let hp = self.observation_matrix.mat_mul(&self.covariance);
        let hpht = hp.mat_mul(&self.observation_matrix.transpose());
        let s = hpht.add(&self.measurement_noise);

        // K = P * H' * S^-1 (simplified: assume S is diagonal for inversion)
        let mut s_inv = CovarianceMatrix::zeros(self.dim);
        for i in 0..self.dim {
            if s.data[i][i].abs() > 1e-12 {
                s_inv.data[i][i] = 1.0 / s.data[i][i];
            }
        }
        let pht = self.covariance.mat_mul(&self.observation_matrix.transpose());
        let k = pht.mat_mul(&s_inv);

        // x = x + K * y
        let ky = mat_vec_mul(&k, &y);
        for i in 0..self.dim {
            self.state[i] += ky[i];
        }

        // P = (I - K * H) * P
        let kh = k.mat_mul(&self.observation_matrix);
        let mut i_kh = CovarianceMatrix::identity(self.dim);
        for i in 0..self.dim {
            for j in 0..self.dim {
                i_kh.data[i][j] -= kh.data[i][j];
            }
        }
        self.covariance = i_kh.mat_mul(&self.covariance);
    }

    /// Full predict + update cycle.
    pub fn step(&mut self, observation: &TernaryObservation) {
        self.predict();
        self.update(observation);
    }

    /// Get the state projected to ternary.
    pub fn ternary_state(&self) -> Vec<Ternary> {
        self.state
            .iter()
            .map(|&v| {
                if v > 0.5 {
                    Ternary::Pos
                } else if v < -0.5 {
                    Ternary::Neg
                } else {
                    Ternary::Zero
                }
            })
            .collect()
    }

    /// Smooth a sequence of filtered states (simple backward smoothing).
    pub fn smooth(filtered_states: &[Vec<f64>], filtered_covs: &[CovarianceMatrix]) -> Vec<Vec<f64>> {
        if filtered_states.is_empty() {
            return vec![];
        }
        let n = filtered_states.len();
        let dim = filtered_states[0].len();
        let mut smoothed = filtered_states.to_vec();
        let mut smoothed_covs = filtered_covs.to_vec();

        // Backward pass
        for t in (0..n - 1).rev() {
            // Simplified: just average with next smoothed state
            for i in 0..dim {
                smoothed[t][i] = (smoothed[t][i] + smoothed[t + 1][i]) / 2.0;
            }
        }
        smoothed
    }
}

/// Fuse multiple measurements into a single observation using weighted average.
pub fn fuse_observations(observations: &[TernaryObservation], weights: &[f64]) -> TernaryObservation {
    if observations.is_empty() {
        return TernaryObservation::new(vec![]);
    }
    let len = observations[0].len();
    let mut fused = vec![0.0; len];
    let total_weight: f64 = weights.iter().sum();

    for (obs, &w) in observations.iter().zip(weights.iter()) {
        let fv = obs.to_f64_vec();
        for i in 0..len {
            fused[i] += fv[i] * w;
        }
    }

    let values: Vec<Ternary> = fused
        .iter()
        .map(|&v| {
            let normalized = v / total_weight;
            if normalized > 0.5 {
                Ternary::Pos
            } else if normalized < -0.5 {
                Ternary::Neg
            } else {
                Ternary::Zero
            }
        })
        .collect();

    TernaryObservation::new(values)
}

/// RTS smoother for ternary Kalman filter (Rauch-Tung-Striebel simplified).
pub fn rts_smooth(
    filtered_states: &[Vec<f64>],
    filtered_covs: &[CovarianceMatrix],
    transition: &CovarianceMatrix,
) -> Vec<Vec<f64>> {
    if filtered_states.is_empty() {
        return vec![];
    }
    let n = filtered_states.len();
    let dim = filtered_states[0].len();
    let mut smoothed = filtered_states.to_vec();
    let mut smoothed_covs = filtered_covs.to_vec();

    for t in (0..n - 1).rev() {
        let p_pred = transition
            .mat_mul(&smoothed_covs[t])
            .mat_mul(&transition.transpose());

        // Simplified gain: just use identity scaling
        for i in 0..dim {
            smoothed[t][i] = filtered_states[t][i]
                + 0.5 * (smoothed[t + 1][i] - filtered_states[t][i]);
        }
    }
    smoothed
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ternary_from_i8() {
        assert_eq!(Ternary::from_i8(-1), Some(Ternary::Neg));
        assert_eq!(Ternary::from_i8(0), Some(Ternary::Zero));
        assert_eq!(Ternary::from_i8(1), Some(Ternary::Pos));
        assert_eq!(Ternary::from_i8(2), None);
    }

    #[test]
    fn test_ternary_to_f64() {
        assert_eq!(Ternary::Neg.to_f64(), -1.0);
        assert_eq!(Ternary::Zero.to_f64(), 0.0);
        assert_eq!(Ternary::Pos.to_f64(), 1.0);
    }

    #[test]
    fn test_fixed_point_add() {
        let a = FixedPoint::from_f64(1.5);
        let b = FixedPoint::from_f64(2.5);
        let c = a.add(b);
        assert!((c.to_f64() - 4.0).abs() < 0.01);
    }

    #[test]
    fn test_fixed_point_mul() {
        let a = FixedPoint::from_f64(3.0);
        let b = FixedPoint::from_f64(2.0);
        let c = a.mul(b);
        assert!((c.to_f64() - 6.0).abs() < 0.01);
    }

    #[test]
    fn test_fixed_point_div() {
        let a = FixedPoint::from_f64(6.0);
        let b = FixedPoint::from_f64(3.0);
        let c = a.div(b);
        assert!((c.to_f64() - 2.0).abs() < 0.01);
    }

    #[test]
    fn test_fixed_point_sub() {
        let a = FixedPoint::from_f64(5.0);
        let b = FixedPoint::from_f64(3.0);
        let c = a.sub(b);
        assert!((c.to_f64() - 2.0).abs() < 0.01);
    }

    #[test]
    fn test_covariance_identity() {
        let m = CovarianceMatrix::identity(3);
        assert_eq!(m.get(0, 0), 1.0);
        assert_eq!(m.get(0, 1), 0.0);
        assert_eq!(m.get(2, 2), 1.0);
    }

    #[test]
    fn test_covariance_trace() {
        let m = CovarianceMatrix::identity(4);
        assert_eq!(m.trace(), 4.0);
    }

    #[test]
    fn test_covariance_transpose() {
        let mut m = CovarianceMatrix::zeros(2);
        m.set(0, 1, 3.0);
        let t = m.transpose();
        assert_eq!(t.get(1, 0), 3.0);
        assert_eq!(t.get(0, 1), 0.0);
    }

    #[test]
    fn test_covariance_mat_mul() {
        let mut a = CovarianceMatrix::identity(2);
        a.set(0, 1, 2.0);
        let b = CovarianceMatrix::identity(2);
        let c = a.mat_mul(&b);
        assert_eq!(c.get(0, 1), 2.0);
    }

    #[test]
    fn test_covariance_scale() {
        let m = CovarianceMatrix::identity(2);
        let s = m.scale(3.0);
        assert_eq!(s.get(0, 0), 3.0);
        assert_eq!(s.get(1, 1), 3.0);
    }

    #[test]
    fn test_observation_to_f64() {
        let obs = TernaryObservation::new(vec![Ternary::Pos, Ternary::Neg, Ternary::Zero]);
        assert_eq!(obs.to_f64_vec(), vec![1.0, -1.0, 0.0]);
    }

    #[test]
    fn test_state_estimator_projection() {
        let se = StateEstimator::new(
            vec![0.8, -0.7, 0.1],
            CovarianceMatrix::identity(3),
        );
        let proj = se.ternary_projection();
        assert_eq!(proj, vec![Ternary::Pos, Ternary::Neg, Ternary::Zero]);
    }

    #[test]
    fn test_kalman_predict() {
        let mut kf = TernaryKalmanFilter::new(2);
        kf.state = vec![1.0, -1.0];
        kf.predict();
        // With identity transition, state should stay same
        assert!((kf.state[0] - 1.0).abs() < 0.01);
        assert!((kf.state[1] - (-1.0)).abs() < 0.01);
    }

    #[test]
    fn test_kalman_update() {
        let mut kf = TernaryKalmanFilter::new(2);
        kf.state = vec![0.0, 0.0];
        let obs = TernaryObservation::new(vec![Ternary::Pos, Ternary::Pos]);
        kf.update(&obs);
        // State should move toward observation
        assert!(kf.state[0] > 0.0);
        assert!(kf.state[1] > 0.0);
    }

    #[test]
    fn test_kalman_step() {
        let mut kf = TernaryKalmanFilter::new(3);
        let obs = TernaryObservation::new(vec![Ternary::Pos, Ternary::Zero, Ternary::Neg]);
        kf.step(&obs);
        assert!(kf.state[0] > 0.0);
    }

    #[test]
    fn test_kalman_ternary_state() {
        let mut kf = TernaryKalmanFilter::new(2);
        kf.state = vec![0.8, -0.7];
        let ts = kf.ternary_state();
        assert_eq!(ts[0], Ternary::Pos);
        assert_eq!(ts[1], Ternary::Neg);
    }

    #[test]
    fn test_fuse_observations() {
        let obs1 = TernaryObservation::new(vec![Ternary::Pos]);
        let obs2 = TernaryObservation::new(vec![Ternary::Pos]);
        let fused = fuse_observations(&[obs1, obs2], &[1.0, 1.0]);
        assert_eq!(fused.values[0], Ternary::Pos);
    }

    #[test]
    fn test_fuse_mixed_observations() {
        let obs1 = TernaryObservation::new(vec![Ternary::Pos]);
        let obs2 = TernaryObservation::new(vec![Ternary::Neg]);
        let fused = fuse_observations(&[obs1, obs2], &[1.0, 1.0]);
        // (1*1 + 1*(-1))/2 = 0 => Zero
        assert_eq!(fused.values[0], Ternary::Zero);
    }

    #[test]
    fn test_fuse_empty() {
        let fused = fuse_observations(&[], &[]);
        assert!(fused.is_empty());
    }

    #[test]
    fn test_smooth_empty() {
        let result = TernaryKalmanFilter::smooth(&[], &[]);
        assert!(result.is_empty());
    }

    #[test]
    fn test_smooth_single() {
        let states = vec![vec![1.0, 0.0]];
        let covs = vec![CovarianceMatrix::identity(2)];
        let result = TernaryKalmanFilter::smooth(&states, &covs);
        assert_eq!(result.len(), 1);
        assert_eq!(result[0][0], 1.0);
    }

    #[test]
    fn test_smooth_multi() {
        let states = vec![vec![1.0], vec![1.0], vec![1.0]];
        let covs = vec![
            CovarianceMatrix::identity(1),
            CovarianceMatrix::identity(1),
            CovarianceMatrix::identity(1),
        ];
        let result = TernaryKalmanFilter::smooth(&states, &covs);
        assert_eq!(result.len(), 3);
    }

    #[test]
    fn test_mat_vec_mul() {
        let m = CovarianceMatrix::identity(2);
        let v = vec![3.0, 4.0];
        let result = mat_vec_mul(&m, &v);
        assert_eq!(result, vec![3.0, 4.0]);
    }

    #[test]
    fn test_rts_smooth() {
        let states = vec![vec![1.0], vec![0.5], vec![0.0]];
        let covs = vec![
            CovarianceMatrix::identity(1),
            CovarianceMatrix::identity(1),
            CovarianceMatrix::identity(1),
        ];
        let transition = CovarianceMatrix::identity(1);
        let result = rts_smooth(&states, &covs, &transition);
        assert_eq!(result.len(), 3);
    }

    #[test]
    fn test_convergence() {
        let mut kf = TernaryKalmanFilter::new(1);
        for _ in 0..20 {
            let obs = TernaryObservation::new(vec![Ternary::Pos]);
            kf.step(&obs);
        }
        assert!(kf.state[0] > 0.8);
    }
}
