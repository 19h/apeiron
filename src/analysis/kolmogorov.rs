//! Kolmogorov complexity approximation using DEFLATE compression.
//!
//! Kolmogorov complexity K(x) is the length of the shortest program that produces x.
//! Since K(x) is uncomputable, we approximate it using the compression ratio.

use flate2::write::DeflateEncoder;
use flate2::Compression;
use std::io::Write;

/// Approximate Kolmogorov complexity using DEFLATE compression.
///
/// Returns a value between 0.0 (highly compressible/simple) and 1.0 (incompressible/random).
/// Values > 1.0 are clamped (can occur with very small inputs due to compression overhead).
pub fn calculate_kolmogorov_complexity(data: &[u8]) -> f64 {
    if data.is_empty() {
        return 0.0;
    }

    // Use maximum compression level for best approximation
    let mut encoder = DeflateEncoder::new(Vec::new(), Compression::best());
    if encoder.write_all(data).is_err() {
        return 1.0; // Assume incompressible on error
    }

    let compressed = match encoder.finish() {
        Ok(c) => c,
        Err(_) => return 1.0,
    };

    // Compression ratio as complexity approximation
    // 0.0 = highly compressible (simple), 1.0 = incompressible (random/complex)
    let ratio = compressed.len() as f64 / data.len() as f64;

    // Clamp to [0, 1] - ratios > 1 can occur with small inputs or already compressed data
    ratio.clamp(0.0, 1.0)
}

/// Kolmogorov complexity analysis results for color mapping.
#[derive(Debug, Clone, Copy)]
pub struct KolmogorovAnalysis {
    /// Compression ratio (0.0 = highly compressible, 1.0 = incompressible).
    pub complexity: f64,
}

impl KolmogorovAnalysis {
    /// Create a new analysis from raw data.
    pub fn from_data(data: &[u8]) -> Self {
        Self {
            complexity: calculate_kolmogorov_complexity(data),
        }
    }

    /// Map Kolmogorov complexity to RGB color.
    ///
    /// Color legend (viridis-inspired for scientific data):
    /// - Deep purple/blue: Very low complexity (simple/repetitive data)
    /// - Teal/green: Low-medium complexity (structured data)
    /// - Yellow: Medium-high complexity (mixed/code)
    /// - Orange/red: High complexity (compressed/encrypted)
    /// - Bright pink/white: Maximum complexity (random/noise)
    pub fn to_color(&self) -> [u8; 3] {
        let t = self.complexity.clamp(0.0, 1.0);

        // Viridis-inspired colormap for scientific visualization
        let (r, g, b) = if t < 0.2 {
            // Deep purple to blue
            let s = t / 0.2;
            (0.25 + s * 0.05, 0.0 + s * 0.15, 0.5 + s * 0.2)
        } else if t < 0.4 {
            // Blue to teal
            let s = (t - 0.2) / 0.2;
            (0.3 - s * 0.15, 0.15 + s * 0.4, 0.7 - s * 0.1)
        } else if t < 0.6 {
            // Teal to green-yellow
            let s = (t - 0.4) / 0.2;
            (0.15 + s * 0.6, 0.55 + s * 0.25, 0.6 - s * 0.4)
        } else if t < 0.8 {
            // Yellow to orange
            let s = (t - 0.6) / 0.2;
            (0.75 + s * 0.25, 0.8 - s * 0.3, 0.2 - s * 0.1)
        } else {
            // Orange to hot pink/red
            let s = (t - 0.8) / 0.2;
            (1.0, 0.5 - s * 0.3, 0.1 + s * 0.4)
        };

        [(r * 255.0) as u8, (g * 255.0) as u8, (b * 255.0) as u8]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_complexity_uniform() {
        // All zeros should be highly compressible
        let data = vec![0u8; 1000];
        let complexity = calculate_kolmogorov_complexity(&data);
        assert!(
            complexity < 0.1,
            "Uniform data should have low complexity: {}",
            complexity
        );
    }

    #[test]
    fn test_complexity_random() {
        // "Random" data should be less compressible
        let data: Vec<u8> = (0..1000).map(|i| (i * 17 + 31) as u8).collect();
        let complexity = calculate_kolmogorov_complexity(&data);
        assert!(
            complexity > 0.3,
            "Varied data should have higher complexity: {}",
            complexity
        );
    }

    #[test]
    fn test_analysis_color() {
        let low = KolmogorovAnalysis { complexity: 0.0 };
        let high = KolmogorovAnalysis { complexity: 1.0 };

        let low_color = low.to_color();
        let high_color = high.to_color();

        // Low complexity should be more blue
        assert!(low_color[2] > low_color[0]);
        // High complexity should be more red
        assert!(high_color[0] > high_color[2]);
    }
}
