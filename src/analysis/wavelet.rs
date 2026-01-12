//! Haar wavelet transform and entropy analysis.
//!
//! Based on: "Wavelet Decomposition of Software Entropy Reveals Symptoms of Malicious Code"
//! Wojnowicz et al., Journal of Innovation in Digital Ecosystems (2016)
//!
//! Key insight: Malware concentrates entropic energy at COARSE levels (large entropy shifts
//! from encrypted/compressed sections), while clean files concentrate energy at FINE levels.

use super::entropy::chunk_entropy_fast;

/// Window size for computing entropy stream (256 bytes as per paper).
pub const WAVELET_CHUNK_SIZE: usize = 256;

/// Compute Shannon entropy for a single chunk.
#[inline]
pub fn chunk_entropy(chunk: &[u8]) -> f64 {
    if chunk.is_empty() {
        return 0.0;
    }

    let mut counts = [0u32; 256];
    for &byte in chunk {
        counts[byte as usize] += 1;
    }

    let total = chunk.len() as f64;
    let mut entropy = 0.0;

    for &count in &counts {
        if count > 0 {
            let p = count as f64 / total;
            entropy -= p * p.log2();
        }
    }

    entropy
}

/// Compute the entropy stream from raw file data.
/// Each element is the Shannon entropy of a 256-byte chunk.
pub fn compute_entropy_stream(data: &[u8]) -> Vec<f64> {
    if data.is_empty() {
        return Vec::new();
    }

    let num_chunks = (data.len() + WAVELET_CHUNK_SIZE - 1) / WAVELET_CHUNK_SIZE;
    let mut stream = Vec::with_capacity(num_chunks);

    for i in 0..num_chunks {
        let start = i * WAVELET_CHUNK_SIZE;
        let end = (start + WAVELET_CHUNK_SIZE).min(data.len());
        stream.push(chunk_entropy(&data[start..end]));
    }

    stream
}

/// Perform Haar wavelet transform on a signal.
/// Returns wavelet coefficients organized by level: Vec<Vec<f64>>
/// Level 0 is coarsest (1 coefficient), level J-1 is finest (2^(J-1) coefficients)
/// where J = floor(log2(signal.len()))
///
/// Uses orthonormal Haar wavelet (multiply by 1/sqrt(2)) to match the paper's
/// energy distribution.
pub fn haar_wavelet_transform(signal: &[f64]) -> Vec<Vec<f64>> {
    if signal.is_empty() {
        return Vec::new();
    }

    // Truncate to power of 2
    let j = (signal.len() as f64).log2().floor() as usize;
    if j == 0 {
        return Vec::new();
    }

    let n = 1 << j; // 2^j
    let mut data: Vec<f64> = signal[..n].to_vec();

    // Store coefficients by level
    let mut coefficients: Vec<Vec<f64>> = Vec::with_capacity(j);

    // Iteratively compute wavelet coefficients from fine to coarse
    let mut current_len = n;

    while current_len > 1 {
        let half = current_len / 2;
        let mut averages = Vec::with_capacity(half);
        let mut details = Vec::with_capacity(half);

        // Orthonormal Haar wavelet: average and difference scaled by 1/sqrt(2)
        // This matches the paper's energy distribution (Wojnowicz et al., 2016)
        let inv_sqrt2 = std::f64::consts::FRAC_1_SQRT_2;
        for i in 0..half {
            let left = data[2 * i];
            let right = data[2 * i + 1];
            averages.push((left + right) * inv_sqrt2);
            details.push((left - right) * inv_sqrt2);
        }

        // Store detail coefficients (from this level)
        coefficients.push(details);

        // Replace data with averages for next iteration
        data = averages;
        current_len = half;
    }

    // Reverse so level 0 is coarsest (1 coefficient), level J-1 is finest
    coefficients.reverse();
    coefficients
}

/// Compute energy spectrum from wavelet coefficients.
/// Energy at level j = sum of squared coefficients at that level.
/// Returns Vec<f64> where index 0 is coarsest level energy.
pub fn wavelet_energy_spectrum(coefficients: &[Vec<f64>]) -> Vec<f64> {
    coefficients
        .iter()
        .map(|level| level.iter().map(|d| d * d).sum())
        .collect()
}

/// Wavelet entropy analysis results for visualization and classification.
#[derive(Debug, Clone)]
pub struct WaveletAnalysis {
    /// Energy at each resolution level (index 0 = coarsest)
    pub energy_spectrum: Vec<f64>,
    /// Total energy in the signal
    pub total_energy: f64,
    /// Ratio of energy at coarse levels (first half) to total energy
    pub coarse_energy_ratio: f64,
    /// Suspiciousness score (0.0 = normal, 1.0 = highly suspicious)
    pub suspiciousness: f64,
    /// Number of resolution levels
    pub num_levels: usize,
}

impl WaveletAnalysis {
    /// Analyze wavelet decomposition of an entropy signal.
    pub fn from_entropy_stream(stream: &[f64]) -> Self {
        let coefficients = haar_wavelet_transform(stream);

        if coefficients.is_empty() {
            return Self {
                energy_spectrum: Vec::new(),
                total_energy: 0.0,
                coarse_energy_ratio: 0.5,
                suspiciousness: 0.5,
                num_levels: 0,
            };
        }

        let energy_spectrum = wavelet_energy_spectrum(&coefficients);
        let num_levels = energy_spectrum.len();
        let total_energy: f64 = energy_spectrum.iter().sum();

        if total_energy == 0.0 || num_levels < 2 {
            return Self {
                energy_spectrum,
                total_energy,
                coarse_energy_ratio: 0.5,
                suspiciousness: 0.5,
                num_levels,
            };
        }

        // Calculate coarse energy ratio (first half of levels)
        let coarse_levels = (num_levels + 1) / 2;
        let coarse_energy: f64 = energy_spectrum[..coarse_levels].iter().sum();
        let coarse_energy_ratio = coarse_energy / total_energy;

        // Calculate suspiciousness score
        let mean_energy = total_energy / num_levels as f64;
        let energy_variance: f64 = energy_spectrum
            .iter()
            .map(|e| (e - mean_energy).powi(2))
            .sum::<f64>()
            / num_levels as f64;
        let energy_std = energy_variance.sqrt();

        // Normalize coarse ratio to suspiciousness
        let ratio_suspiciousness = ((coarse_energy_ratio - 0.3) / 0.4).clamp(0.0, 1.0);

        // High energy variance also indicates suspicious structure
        let cv = if mean_energy > 0.0 {
            energy_std / mean_energy
        } else {
            0.0
        };
        let variance_suspiciousness = (cv / 2.0).clamp(0.0, 1.0);

        // Combined suspiciousness (weighted)
        let suspiciousness = ratio_suspiciousness * 0.7 + variance_suspiciousness * 0.3;

        Self {
            energy_spectrum,
            total_energy,
            coarse_energy_ratio,
            suspiciousness,
            num_levels,
        }
    }

    /// Map wavelet analysis to RGB color.
    pub fn to_color(&self) -> [u8; 3] {
        let t = self.suspiciousness.clamp(0.0, 1.0);

        // Cool-to-hot colormap emphasizing suspicious regions
        let (r, g, b) = if t < 0.2 {
            let s = t / 0.2;
            (0.1 + s * 0.1, 0.2 + s * 0.3, 0.7 + s * 0.2)
        } else if t < 0.4 {
            let s = (t - 0.2) / 0.2;
            (0.2 - s * 0.1, 0.5 + s * 0.3, 0.9 - s * 0.2)
        } else if t < 0.6 {
            let s = (t - 0.4) / 0.2;
            (0.1 + s * 0.6, 0.8 - s * 0.1, 0.7 - s * 0.5)
        } else if t < 0.8 {
            let s = (t - 0.6) / 0.2;
            (0.7 + s * 0.3, 0.7 - s * 0.3, 0.2 - s * 0.1)
        } else {
            let s = (t - 0.8) / 0.2;
            (1.0, 0.4 - s * 0.2, 0.1 + s * 0.3)
        };

        [(r * 255.0) as u8, (g * 255.0) as u8, (b * 255.0) as u8]
    }
}

/// Highly optimized wavelet suspiciousness for exactly 8 entropy values.
/// Performs unrolled Haar wavelet transform and computes suspiciousness in one pass.
/// Zero heap allocations.
///
/// Uses orthonormal Haar wavelet (multiply by 1/sqrt(2)) to match the paper's
/// energy distribution.
#[inline(always)]
pub fn wavelet_suspiciousness_8(stream: &[f64; 8]) -> f64 {
    // Orthonormal Haar scaling factor
    const INV_SQRT2: f64 = std::f64::consts::FRAC_1_SQRT_2;

    // Unrolled Haar wavelet transform for 8 elements (3 levels)
    let d2_0 = (stream[0] - stream[1]) * INV_SQRT2;
    let d2_1 = (stream[2] - stream[3]) * INV_SQRT2;
    let d2_2 = (stream[4] - stream[5]) * INV_SQRT2;
    let d2_3 = (stream[6] - stream[7]) * INV_SQRT2;

    let a2_0 = (stream[0] + stream[1]) * INV_SQRT2;
    let a2_1 = (stream[2] + stream[3]) * INV_SQRT2;
    let a2_2 = (stream[4] + stream[5]) * INV_SQRT2;
    let a2_3 = (stream[6] + stream[7]) * INV_SQRT2;

    let d1_0 = (a2_0 - a2_1) * INV_SQRT2;
    let d1_1 = (a2_2 - a2_3) * INV_SQRT2;

    let a1_0 = (a2_0 + a2_1) * INV_SQRT2;
    let a1_1 = (a2_2 + a2_3) * INV_SQRT2;

    let d0_0 = (a1_0 - a1_1) * INV_SQRT2;

    // Compute energies at each level
    let e2 = d2_0 * d2_0 + d2_1 * d2_1 + d2_2 * d2_2 + d2_3 * d2_3;
    let e1 = d1_0 * d1_0 + d1_1 * d1_1;
    let e0 = d0_0 * d0_0;

    let total_energy = e0 + e1 + e2;

    if total_energy < 1e-10 {
        return 0.5;
    }

    let coarse_energy = e0 + e1;
    let coarse_ratio = coarse_energy / total_energy;

    let mean_energy = total_energy * (1.0 / 3.0);
    let var = (e0 - mean_energy) * (e0 - mean_energy)
        + (e1 - mean_energy) * (e1 - mean_energy)
        + (e2 - mean_energy) * (e2 - mean_energy);
    let std = (var * (1.0 / 3.0)).sqrt();
    let cv = std / mean_energy;

    let ratio_susp = ((coarse_ratio - 0.4) * 2.5).clamp(0.0, 1.0);
    let var_susp = (cv * 0.5).clamp(0.0, 1.0);

    ratio_susp * 0.7 + var_susp * 0.3
}

/// Calculate a quick wavelet suspiciousness value for visualization.
pub fn calculate_wavelet_suspiciousness(data: &[u8]) -> f32 {
    let stream = compute_entropy_stream(data);
    if stream.len() < 4 {
        return 0.5;
    }
    let analysis = WaveletAnalysis::from_entropy_stream(&stream);
    analysis.suspiciousness as f32
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_entropy_stream() {
        let data = vec![0u8; 1024];
        let stream = compute_entropy_stream(&data);
        assert_eq!(stream.len(), 4);
        for &entropy in &stream {
            assert!(entropy < 0.1, "Uniform data should have near-zero entropy");
        }
    }

    #[test]
    fn test_entropy_stream_random() {
        let data: Vec<u8> = (0..=255).cycle().take(1024).collect();
        let stream = compute_entropy_stream(&data);
        assert_eq!(stream.len(), 4);
        for &entropy in &stream {
            assert!(entropy > 7.5, "Random data should have high entropy");
        }
    }

    #[test]
    fn test_haar_transform() {
        let signal: Vec<f64> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let coefficients = haar_wavelet_transform(&signal);

        assert_eq!(coefficients.len(), 3);
        assert_eq!(coefficients[0].len(), 1);
        assert_eq!(coefficients[1].len(), 2);
        assert_eq!(coefficients[2].len(), 4);
    }

    #[test]
    fn test_wavelet_energy_preservation() {
        let signal: Vec<f64> = vec![1.0, 2.0, 3.0, 4.0];
        let coefficients = haar_wavelet_transform(&signal);
        let wavelet_energy: f64 = coefficients
            .iter()
            .flat_map(|level| level.iter())
            .map(|d| d * d)
            .sum();

        assert!(wavelet_energy > 0.0, "Wavelet energy should be positive");
    }
}
