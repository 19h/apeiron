//! Jensen-Shannon Divergence analysis.
//!
//! JSD is a symmetric and bounded measure of similarity between probability distributions.
//! Used to detect anomalous regions in binary files.

use super::entropy::byte_distribution;

/// Calculate KL divergence D_KL(P || Q) between two probability distributions.
///
/// Uses a small epsilon to avoid log(0) issues.
fn kl_divergence(p: &[f64; 256], q: &[f64; 256]) -> f64 {
    const EPSILON: f64 = 1e-10;
    let mut divergence = 0.0;

    for i in 0..256 {
        let p_i = p[i];
        let q_i = q[i];

        if p_i > EPSILON {
            // Add epsilon to q to avoid log(0)
            divergence += p_i * (p_i / (q_i + EPSILON)).ln();
        }
    }

    divergence / std::f64::consts::LN_2 // Convert to bits (log base 2)
}

/// Calculate Jensen-Shannon divergence between two probability distributions.
///
/// JSD is a symmetric and bounded measure of similarity between distributions.
/// JSD(P || Q) = 0.5 * KL(P || M) + 0.5 * KL(Q || M), where M = 0.5 * (P + Q)
///
/// Returns a value between 0.0 (identical distributions) and 1.0 (maximally different).
pub fn jensen_shannon_divergence(p: &[f64; 256], q: &[f64; 256]) -> f64 {
    // Calculate M = 0.5 * (P + Q)
    let mut m = [0.0f64; 256];
    for i in 0..256 {
        m[i] = 0.5 * (p[i] + q[i]);
    }

    // JSD = 0.5 * KL(P || M) + 0.5 * KL(Q || M)
    let jsd = 0.5 * kl_divergence(p, &m) + 0.5 * kl_divergence(q, &m);

    // JSD is bounded [0, 1] when using log base 2
    jsd.clamp(0.0, 1.0)
}

/// Calculate Jensen-Shannon divergence of a data chunk against a reference distribution.
///
/// Returns a value between 0.0 (identical to reference) and 1.0 (maximally different).
pub fn calculate_jsd(data: &[u8], reference: &[f64; 256]) -> f64 {
    let local_dist = byte_distribution(data);
    jensen_shannon_divergence(&local_dist, reference)
}

/// Jensen-Shannon divergence analysis results for color mapping.
#[derive(Debug, Clone, Copy)]
pub struct JSDAnalysis {
    /// JSD value (0.0 = similar to reference, 1.0 = very different).
    pub divergence: f64,
}

impl JSDAnalysis {
    /// Create a new analysis from data and reference distribution.
    pub fn from_data(data: &[u8], reference: &[f64; 256]) -> Self {
        Self {
            divergence: calculate_jsd(data, reference),
        }
    }

    /// Map JSD to RGB color.
    ///
    /// Color legend (divergence from reference):
    /// - Dark blue/purple: Very similar to reference (low divergence)
    /// - Cyan/teal: Slightly different
    /// - Green/yellow: Moderately different
    /// - Orange: Significantly different
    /// - Red/magenta: Very different (high divergence/anomalous)
    pub fn to_color(&self) -> [u8; 3] {
        let t = self.divergence.clamp(0.0, 1.0);

        // Cool-to-hot colormap emphasizing anomalies
        let (r, g, b) = if t < 0.15 {
            // Dark blue - very similar
            let s = t / 0.15;
            (0.1, 0.1 + s * 0.2, 0.4 + s * 0.3)
        } else if t < 0.3 {
            // Blue to cyan
            let s = (t - 0.15) / 0.15;
            (0.1, 0.3 + s * 0.4, 0.7 - s * 0.1)
        } else if t < 0.5 {
            // Cyan to green
            let s = (t - 0.3) / 0.2;
            (0.1 + s * 0.3, 0.7 - s * 0.1, 0.6 - s * 0.4)
        } else if t < 0.7 {
            // Green to yellow
            let s = (t - 0.5) / 0.2;
            (0.4 + s * 0.6, 0.6 + s * 0.3, 0.2 - s * 0.1)
        } else if t < 0.85 {
            // Yellow to orange
            let s = (t - 0.7) / 0.15;
            (1.0, 0.9 - s * 0.4, 0.1)
        } else {
            // Orange to red/magenta - anomalous
            let s = (t - 0.85) / 0.15;
            (1.0, 0.5 - s * 0.3, 0.1 + s * 0.4)
        };

        [(r * 255.0) as u8, (g * 255.0) as u8, (b * 255.0) as u8]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_jsd_identical() {
        let dist = [1.0 / 256.0; 256];
        let jsd = jensen_shannon_divergence(&dist, &dist);
        assert!(jsd < 0.01, "Identical distributions should have JSD near 0");
    }

    #[test]
    fn test_jsd_different() {
        let mut dist1 = [0.0f64; 256];
        let mut dist2 = [0.0f64; 256];

        // Concentrate all probability mass at different bytes
        dist1[0] = 1.0;
        dist2[255] = 1.0;

        let jsd = jensen_shannon_divergence(&dist1, &dist2);
        assert!(
            jsd > 0.9,
            "Completely different distributions should have JSD near 1"
        );
    }

    #[test]
    fn test_jsd_symmetric() {
        let dist1 = [1.0 / 256.0; 256];
        let mut dist2 = [0.0f64; 256];
        for i in 0..256 {
            dist2[i] = if i < 128 { 2.0 / 256.0 } else { 0.0 };
        }

        let jsd1 = jensen_shannon_divergence(&dist1, &dist2);
        let jsd2 = jensen_shannon_divergence(&dist2, &dist1);

        assert!((jsd1 - jsd2).abs() < 0.001, "JSD should be symmetric");
    }
}
