// Allow unused code - this module provides a library of analysis utilities,
// not all of which are used in every visualization mode.
#![allow(dead_code)]

//! Analysis utilities for binary data inspection.
//!
//! Provides entropy calculation, hex dump formatting, file type detection,
//! Kolmogorov complexity approximation, and forensic color mapping for visualization.

use flate2::write::DeflateEncoder;
use flate2::Compression;
use std::fmt::Write;
use std::io::Write as IoWrite;

/// Window size for byte analysis (matches original implementation).
pub const ANALYSIS_WINDOW: usize = 64;

/// Calculate Shannon entropy for a byte slice.
///
/// Shannon entropy measures the average information content per byte.
/// Values range from 0 (completely uniform) to 8 (maximum randomness).
pub fn calculate_entropy(data: &[u8]) -> f64 {
    if data.is_empty() {
        return 0.0;
    }

    let mut counts = [0u32; 256];
    for &byte in data {
        counts[byte as usize] += 1;
    }

    let total = data.len() as f64;
    let mut entropy = 0.0;

    for &count in &counts {
        if count > 0 {
            let p = count as f64 / total;
            entropy -= p * p.log2();
        }
    }

    entropy
}

/// Approximate Kolmogorov complexity using DEFLATE compression.
///
/// Kolmogorov complexity K(x) is the length of the shortest program that produces x.
/// Since K(x) is uncomputable, we approximate it using the compression ratio:
/// K(x) ≈ compressed_length / original_length
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

// =============================================================================
// Jensen-Shannon Divergence
// =============================================================================

/// Calculate byte frequency distribution (probability distribution) for a byte slice.
///
/// Returns an array of 256 probabilities, one for each possible byte value.
pub fn byte_distribution(data: &[u8]) -> [f64; 256] {
    if data.is_empty() {
        // Return uniform distribution for empty data
        return [1.0 / 256.0; 256];
    }

    let mut counts = [0u64; 256];
    for &byte in data {
        counts[byte as usize] += 1;
    }

    let total = data.len() as f64;
    let mut dist = [0.0f64; 256];
    for i in 0..256 {
        dist[i] = counts[i] as f64 / total;
    }
    dist
}

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

// =============================================================================
// Refined Composite Multi-Scale Entropy (RCMSE) - Optimized Implementation
// =============================================================================

use rayon::prelude::*;
use std::sync::atomic::{AtomicU64, Ordering};

/// Default parameters for RCMSE calculation.
pub const RCMSE_EMBEDDING_DIM: usize = 2; // m parameter
pub const RCMSE_TOLERANCE_FACTOR: f64 = 0.15; // r = 0.15 * SD
pub const RCMSE_MAX_SCALE: usize = 20; // Maximum scale factor τ

/// Coarse-grain a time series at scale τ with offset k, writing into a pre-allocated buffer.
/// Returns the number of points written.
#[inline]
fn coarse_grain_into(data: &[f64], scale: usize, offset: usize, out: &mut [f64]) -> usize {
    if scale == 0 || data.is_empty() || offset == 0 || offset > scale {
        return 0;
    }

    let n = data.len();
    let num_points = (n.saturating_sub(offset - 1)) / scale;

    if num_points == 0 {
        return 0;
    }

    let inv_scale = 1.0 / scale as f64;
    let actual_points = num_points.min(out.len());

    for j in 0..actual_points {
        let start = j * scale + (offset - 1);
        let end = (start + scale).min(n);
        let mut sum = 0.0;
        for i in start..end {
            sum += data[i];
        }
        out[j] = sum * inv_scale;
    }

    actual_points
}

/// Optimized pattern matching using sorted indices and bucket pruning.
/// This is O(n * k) where k is average matches per point, much better than O(n²) for typical data.
///
/// Key optimizations:
/// 1. Sort vectors by first element for cache-friendly access and early termination
/// 2. Only compare vectors within tolerance range (bucket pruning)
/// 3. Unrolled comparison for m=2 (most common case)
/// 4. Process in parallel chunks
#[inline]
fn count_pattern_matches_optimized(series: &[f64], r: f64) -> (u64, u64) {
    let n = series.len();
    const M: usize = 2; // Embedding dimension (hardcoded for optimization)

    if n <= M + 1 {
        return (0, 0);
    }

    let num_templates = n - M;
    let num_templates_extended = n - M - 1;

    // Create index array sorted by first element of each template vector
    let mut indices: Vec<usize> = (0..num_templates).collect();
    indices.sort_unstable_by(|&a, &b| {
        series[a]
            .partial_cmp(&series[b])
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    // For small series, use simple O(n²) which has less overhead
    if num_templates < 32 {
        return count_pattern_matches_simple(series, r);
    }

    // Parallel counting with atomic counters
    let count_m = AtomicU64::new(0);
    let count_m_plus_1 = AtomicU64::new(0);

    // Process in parallel chunks
    indices.par_chunks(64).for_each(|chunk| {
        let mut local_m: u64 = 0;
        let mut local_m_plus_1: u64 = 0;

        for &i in chunk {
            let vi0 = series[i];
            let vi1 = series[i + 1];

            // Binary search to find starting point within tolerance
            let target_min = vi0 - r;
            let target_max = vi0 + r;

            // Find range of indices where first element is within [vi0 - r, vi0 + r]
            let start_idx = indices.partition_point(|&idx| series[idx] < target_min);
            let end_idx = indices.partition_point(|&idx| series[idx] <= target_max);

            // Compare only with vectors in range
            for &j in &indices[start_idx..end_idx] {
                if j <= i {
                    continue; // Avoid double counting and self-comparison
                }

                let vj0 = series[j];
                let vj1 = series[j + 1];

                // Unrolled m=2 Chebyshev distance check
                let d0 = (vi0 - vj0).abs();
                let d1 = (vi1 - vj1).abs();

                if d0 <= r && d1 <= r {
                    local_m += 2; // Count both (i,j) and (j,i)

                    // Check m+1 dimension if both indices are valid
                    if i < num_templates_extended && j < num_templates_extended {
                        let d2 = (series[i + 2] - series[j + 2]).abs();
                        if d2 <= r {
                            local_m_plus_1 += 2;
                        }
                    }
                }
            }
        }

        count_m.fetch_add(local_m, Ordering::Relaxed);
        count_m_plus_1.fetch_add(local_m_plus_1, Ordering::Relaxed);
    });

    (
        count_m.load(Ordering::Relaxed),
        count_m_plus_1.load(Ordering::Relaxed),
    )
}

/// Simple O(n²) pattern matching for small series (less overhead than sorted version).
#[inline]
fn count_pattern_matches_simple(series: &[f64], r: f64) -> (u64, u64) {
    let n = series.len();
    const M: usize = 2;

    if n <= M + 1 {
        return (0, 0);
    }

    let num_templates = n - M;
    let num_templates_extended = n - M - 1;

    let mut count_m: u64 = 0;
    let mut count_m_plus_1: u64 = 0;

    for i in 0..num_templates {
        let vi0 = series[i];
        let vi1 = series[i + 1];

        for j in (i + 1)..num_templates {
            let vj0 = series[j];
            let vj1 = series[j + 1];

            // Unrolled m=2 check with early exit
            let d0 = (vi0 - vj0).abs();
            if d0 > r {
                continue;
            }
            let d1 = (vi1 - vj1).abs();
            if d1 > r {
                continue;
            }

            count_m += 2;

            if i < num_templates_extended && j < num_templates_extended {
                let d2 = (series[i + 2] - series[j + 2]).abs();
                if d2 <= r {
                    count_m_plus_1 += 2;
                }
            }
        }
    }

    (count_m, count_m_plus_1)
}

/// Calculate standard deviation using Welford's online algorithm (numerically stable).
#[inline]
fn std_deviation(data: &[f64]) -> f64 {
    if data.len() < 2 {
        return 0.0;
    }

    let mut mean = 0.0;
    let mut m2 = 0.0;

    for (i, &x) in data.iter().enumerate() {
        let delta = x - mean;
        mean += delta / (i + 1) as f64;
        let delta2 = x - mean;
        m2 += delta * delta2;
    }

    (m2 / (data.len() - 1) as f64).sqrt()
}

/// Calculate RCMSE for a single scale factor τ with optimized coarse-graining.
/// Pre-allocates buffers and processes all offsets efficiently.
fn rcmse_at_scale_optimized(
    data: &[f64],
    scale: usize,
    r: f64,
    buffer: &mut Vec<f64>,
) -> Option<f64> {
    if scale == 0 {
        return None;
    }

    let max_coarse_len = data.len() / scale + 1;
    buffer.resize(max_coarse_len, 0.0);

    let mut total_m: u64 = 0;
    let mut total_m_plus_1: u64 = 0;

    // Process all τ coarse-grained series (offsets 1 to τ)
    for k in 1..=scale {
        let len = coarse_grain_into(data, scale, k, buffer);
        if len <= RCMSE_EMBEDDING_DIM + 1 {
            continue;
        }
        let (n_m, n_m_plus_1) = count_pattern_matches_optimized(&buffer[..len], r);
        total_m += n_m;
        total_m_plus_1 += n_m_plus_1;
    }

    // RCMSE is undefined if no matches found
    if total_m == 0 || total_m_plus_1 == 0 {
        return None;
    }

    let ratio = total_m_plus_1 as f64 / total_m as f64;
    Some(-ratio.ln())
}

/// RCMSE analysis results for visualization.
#[derive(Debug, Clone)]
pub struct RCMSEAnalysis {
    /// Sample entropy values at each scale (scale 1 to max_scale).
    pub entropy_by_scale: Vec<f64>,
    /// Linear regression slope of entropy vs scale.
    /// - Negative slope: entropy decreases with scale (white noise-like, random/encrypted)
    /// - Near-zero slope: entropy constant (1/f noise-like, structured complexity)
    /// - Positive then negative: deterministic chaos pattern
    pub slope: f64,
    /// Complexity index based on the entropy profile.
    /// Higher values indicate more complex, structured data.
    pub complexity_index: f64,
    /// Mean entropy across all scales.
    pub mean_entropy: f64,
    /// Classification based on entropy profile.
    pub classification: RCMSEClassification,
}

/// Classification of data based on RCMSE profile.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RCMSEClassification {
    /// Entropy decreases monotonically (white noise, encrypted, compressed).
    Random,
    /// Entropy is relatively constant across scales (1/f noise, structured complexity).
    Structured,
    /// Entropy increases at small scales then decreases (deterministic chaos).
    Chaotic,
    /// Insufficient data to classify.
    Unknown,
}

impl RCMSEAnalysis {
    /// Map RCMSE analysis to RGB color.
    ///
    /// Color legend based on complexity index and classification:
    /// - Deep blue/purple: Random/encrypted (steep negative slope, high initial entropy)
    /// - Cyan/teal: Compressed data (negative slope, medium complexity)
    /// - Green: Structured/complex (flat slope, 1/f-like)
    /// - Yellow/orange: Chaotic/interesting patterns (positive early slope)
    /// - Red/magenta: Highly structured with low entropy (simple patterns)
    pub fn to_color(&self) -> [u8; 3] {
        // Normalize complexity index to [0, 1]
        let ci = self.complexity_index.clamp(0.0, 2.0) / 2.0;

        // Use slope to determine base hue
        // slope ranges roughly from -0.3 (random) to +0.1 (chaotic)
        let slope_normalized = ((self.slope + 0.3) / 0.4).clamp(0.0, 1.0);

        match self.classification {
            RCMSEClassification::Random => {
                // Purple to blue gradient based on entropy level
                let intensity = (1.0 - ci) * 0.5 + 0.3;
                let r = (0.4 * intensity * 255.0) as u8;
                let g = (0.1 * intensity * 255.0) as u8;
                let b = (0.9 * intensity * 255.0) as u8;
                [r, g, b]
            }
            RCMSEClassification::Structured => {
                // Green to cyan gradient (complex, 1/f-like)
                let intensity = ci * 0.6 + 0.4;
                let r = (0.1 * intensity * 255.0) as u8;
                let g = (0.8 * intensity * 255.0) as u8;
                let b = (0.5 * intensity * 255.0) as u8;
                [r, g, b]
            }
            RCMSEClassification::Chaotic => {
                // Yellow to orange gradient (deterministic chaos)
                let intensity = ci * 0.7 + 0.3;
                let r = (1.0 * intensity * 255.0) as u8;
                let g = (0.7 * intensity * 255.0) as u8;
                let b = (0.1 * intensity * 255.0) as u8;
                [r, g, b]
            }
            RCMSEClassification::Unknown => {
                // Gray for insufficient data
                let v = (0.3 + slope_normalized * 0.4) * 255.0;
                [v as u8, v as u8, v as u8]
            }
        }
    }
}

/// Calculate RCMSE analysis for a data chunk.
///
/// Computes sample entropy at multiple scales and derives complexity metrics.
/// Optimized with parallel scale computation and efficient pattern matching.
pub fn calculate_rcmse(data: &[u8], max_scale: usize) -> RCMSEAnalysis {
    // Convert bytes to f64 for floating-point operations
    let data_f64: Vec<f64> = data.iter().map(|&b| b as f64).collect();

    // Calculate tolerance based on standard deviation
    let sd = std_deviation(&data_f64);
    let r = RCMSE_TOLERANCE_FACTOR * sd;

    // Need minimum data length for meaningful analysis
    // At scale τ, coarse-grained series has length N/τ, need at least m+2 points
    let min_length_for_scale = |scale: usize| (RCMSE_EMBEDDING_DIM + 2) * scale;

    if data.len() < min_length_for_scale(1) || r == 0.0 {
        return RCMSEAnalysis {
            entropy_by_scale: Vec::new(),
            slope: 0.0,
            complexity_index: 0.0,
            mean_entropy: 0.0,
            classification: RCMSEClassification::Unknown,
        };
    }

    // Determine valid scales
    let valid_max_scale = (1..=max_scale)
        .take_while(|&s| data.len() >= min_length_for_scale(s))
        .last()
        .unwrap_or(0);

    if valid_max_scale < 3 {
        return RCMSEAnalysis {
            entropy_by_scale: Vec::new(),
            slope: 0.0,
            complexity_index: 0.0,
            mean_entropy: 0.0,
            classification: RCMSEClassification::Unknown,
        };
    }

    // Parallel computation of entropy at each scale
    let scale_results: Vec<(usize, Option<f64>)> = (1..=valid_max_scale)
        .into_par_iter()
        .map(|scale| {
            // Each thread gets its own buffer
            let mut buffer = Vec::with_capacity(data_f64.len() / scale + 1);
            let entropy = rcmse_at_scale_optimized(&data_f64, scale, r, &mut buffer);
            (scale, entropy)
        })
        .collect();

    // Collect valid results
    let mut entropy_by_scale = Vec::with_capacity(valid_max_scale);
    let mut valid_scales = Vec::with_capacity(valid_max_scale);
    let mut last_valid_entropy = None;

    for (scale, entropy_opt) in scale_results {
        if let Some(entropy) = entropy_opt {
            entropy_by_scale.push(entropy);
            valid_scales.push(scale as f64);
            last_valid_entropy = Some(entropy);
        } else if let Some(prev) = last_valid_entropy {
            // Use previous value for continuity
            entropy_by_scale.push(prev);
            valid_scales.push(scale as f64);
        }
    }

    if entropy_by_scale.len() < 3 {
        return RCMSEAnalysis {
            entropy_by_scale,
            slope: 0.0,
            complexity_index: 0.0,
            mean_entropy: 0.0,
            classification: RCMSEClassification::Unknown,
        };
    }

    // Calculate linear regression slope
    let n = valid_scales.len() as f64;
    let mean_scale: f64 = valid_scales.iter().sum::<f64>() / n;
    let mean_entropy: f64 = entropy_by_scale.iter().sum::<f64>() / n;

    let mut numerator = 0.0;
    let mut denominator = 0.0;
    for (scale, entropy) in valid_scales.iter().zip(entropy_by_scale.iter()) {
        numerator += (scale - mean_scale) * (entropy - mean_entropy);
        denominator += (scale - mean_scale).powi(2);
    }

    let slope = if denominator > 0.0 {
        numerator / denominator
    } else {
        0.0
    };

    // Detect early increase (characteristic of chaotic signals)
    let has_early_increase = entropy_by_scale.len() >= 3
        && entropy_by_scale[1] > entropy_by_scale[0]
        && entropy_by_scale[2] > entropy_by_scale[0];

    // Classification based on slope and pattern
    let classification = if has_early_increase && slope < 0.05 {
        RCMSEClassification::Chaotic
    } else if slope < -0.05 {
        RCMSEClassification::Random
    } else if slope.abs() < 0.05 {
        RCMSEClassification::Structured
    } else {
        RCMSEClassification::Unknown
    };

    // Complexity index: combine mean entropy with slope information
    // Higher complexity for structured (flat slope) with moderate entropy
    let slope_factor = 1.0 - slope.abs().min(0.3) / 0.3; // 1.0 for flat, 0.0 for steep
    let entropy_factor = (mean_entropy / 3.0).min(1.0); // Normalize assuming max ~3.0
    let complexity_index = slope_factor * entropy_factor * 2.0;

    RCMSEAnalysis {
        entropy_by_scale,
        slope,
        complexity_index,
        mean_entropy,
        classification,
    }
}

/// Calculate a simplified RCMSE complexity value for fast visualization.
///
/// Returns a value between 0.0 and 1.0:
/// - 0.0: Random/encrypted (steep negative slope)
/// - 0.5: Structured/complex (flat slope)
/// - 1.0: Chaotic/interesting (positive early slope)
pub fn calculate_rcmse_quick(data: &[u8]) -> f32 {
    // Use fewer scales for speed (6 scales works well with 256-byte windows)
    const QUICK_MAX_SCALE: usize = 6;

    let analysis = calculate_rcmse(data, QUICK_MAX_SCALE);

    // Map to 0-1 based on complexity index and classification
    match analysis.classification {
        RCMSEClassification::Random => 0.1 + analysis.complexity_index as f32 * 0.15,
        RCMSEClassification::Structured => 0.4 + analysis.complexity_index as f32 * 0.25,
        RCMSEClassification::Chaotic => 0.7 + analysis.complexity_index as f32 * 0.2,
        RCMSEClassification::Unknown => 0.5,
    }
}

// =============================================================================
// Wavelet Entropy Decomposition (Haar Wavelet Transform)
// =============================================================================
// Based on: "Wavelet Decomposition of Software Entropy Reveals Symptoms of Malicious Code"
// Wojnowicz et al., Journal of Innovation in Digital Ecosystems (2016)
//
// Key insight: Malware concentrates entropic energy at COARSE levels (large entropy shifts
// from encrypted/compressed sections), while clean files concentrate energy at FINE levels.

/// Window size for computing entropy stream (256 bytes as per paper)
pub const WAVELET_CHUNK_SIZE: usize = 256;

/// Compute Shannon entropy for a single chunk (returns 0-8 bits)
/// Note: Used by tests and the non-optimized path. Production uses chunk_entropy_fast.
#[allow(dead_code)]
#[inline]
fn chunk_entropy(chunk: &[u8]) -> f64 {
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
/// Note: Used by tests and the non-optimized path. Production uses precompute_chunk_entropies.
#[allow(dead_code)]
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
/// Note: Used by tests and WaveletAnalysis. Production uses wavelet_suspiciousness_8.
#[allow(dead_code)]
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

        // Orthonormal Haar wavelet: average and difference scaled by 1/√2
        // This matches the paper's energy distribution (Wojnowicz et al., 2016)
        let inv_sqrt2 = std::f64::consts::FRAC_1_SQRT_2;
        for i in 0..half {
            let left = data[2 * i];
            let right = data[2 * i + 1];
            averages.push((left + right) * inv_sqrt2);
            details.push((left - right) * inv_sqrt2);
        }

        // Store detail coefficients (from this level)
        // We push in reverse order so level 0 is coarsest
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
/// Note: Used by tests and WaveletAnalysis. Production computes energy inline.
#[allow(dead_code)]
pub fn wavelet_energy_spectrum(coefficients: &[Vec<f64>]) -> Vec<f64> {
    coefficients
        .iter()
        .map(|level| level.iter().map(|d| d * d).sum())
        .collect()
}

/// Wavelet entropy analysis results for visualization and classification.
/// Note: Used by tests. Production uses wavelet_suspiciousness_8 directly.
#[allow(dead_code)]
#[derive(Debug, Clone)]
pub struct WaveletAnalysis {
    /// Energy at each resolution level (index 0 = coarsest)
    pub energy_spectrum: Vec<f64>,
    /// Total energy in the signal
    pub total_energy: f64,
    /// Ratio of energy at coarse levels (first half) to total energy
    /// Higher values = more suspicious (malware-like)
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
        // Per the paper: malware concentrates energy at coarse levels
        let coarse_levels = (num_levels + 1) / 2; // Round up for odd numbers
        let coarse_energy: f64 = energy_spectrum[..coarse_levels].iter().sum();
        let coarse_energy_ratio = coarse_energy / total_energy;

        // Calculate suspiciousness score
        // Based on paper findings:
        // - Clean files: energy concentrated at fine levels (low coarse ratio)
        // - Malware: energy concentrated at coarse levels (high coarse ratio)
        //
        // We also consider the energy magnitude and distribution
        let mean_energy = total_energy / num_levels as f64;
        let energy_variance: f64 = energy_spectrum
            .iter()
            .map(|e| (e - mean_energy).powi(2))
            .sum::<f64>()
            / num_levels as f64;
        let energy_std = energy_variance.sqrt();

        // Normalize coarse ratio to suspiciousness
        // Typical clean files: coarse_ratio ~ 0.2-0.4
        // Suspicious files: coarse_ratio ~ 0.5-0.8
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
    ///
    /// Color legend based on suspiciousness:
    /// - Deep blue/cyan: Low suspiciousness (normal, energy at fine levels)
    /// - Green/teal: Medium-low suspiciousness
    /// - Yellow/orange: Medium-high suspiciousness
    /// - Red/magenta: High suspiciousness (malware-like, energy at coarse levels)
    pub fn to_color(&self) -> [u8; 3] {
        let t = self.suspiciousness.clamp(0.0, 1.0);

        // Cool-to-hot colormap emphasizing suspicious regions
        let (r, g, b) = if t < 0.2 {
            // Deep blue - very normal (energy at fine levels)
            let s = t / 0.2;
            (0.1 + s * 0.1, 0.2 + s * 0.3, 0.7 + s * 0.2)
        } else if t < 0.4 {
            // Blue to cyan/teal
            let s = (t - 0.2) / 0.2;
            (0.2 - s * 0.1, 0.5 + s * 0.3, 0.9 - s * 0.2)
        } else if t < 0.6 {
            // Cyan to green/yellow
            let s = (t - 0.4) / 0.2;
            (0.1 + s * 0.6, 0.8 - s * 0.1, 0.7 - s * 0.5)
        } else if t < 0.8 {
            // Yellow to orange
            let s = (t - 0.6) / 0.2;
            (0.7 + s * 0.3, 0.7 - s * 0.3, 0.2 - s * 0.1)
        } else {
            // Orange to red/magenta - highly suspicious
            let s = (t - 0.8) / 0.2;
            (1.0, 0.4 - s * 0.2, 0.1 + s * 0.3)
        };

        [(r * 255.0) as u8, (g * 255.0) as u8, (b * 255.0) as u8]
    }
}

/// Calculate a quick wavelet suspiciousness value for visualization.
/// Returns a value between 0.0 and 1.0.
/// Note: Used by tests. Production uses precompute_chunk_entropies + wavelet_suspiciousness_8.
#[allow(dead_code)]
pub fn calculate_wavelet_suspiciousness(data: &[u8]) -> f32 {
    let stream = compute_entropy_stream(data);
    if stream.len() < 4 {
        return 0.5; // Not enough data for meaningful analysis
    }
    let analysis = WaveletAnalysis::from_entropy_stream(&stream);
    analysis.suspiciousness as f32
}

// =============================================================================
// Optimized Wavelet Analysis (Zero-Allocation, Unrolled)
// =============================================================================

/// Optimized entropy calculation for a 256-byte chunk.
/// Uses precomputed reciprocal and avoids branching where possible.
#[inline(always)]
fn chunk_entropy_fast(chunk: &[u8]) -> f64 {
    debug_assert!(!chunk.is_empty());

    let mut counts = [0u32; 256];
    for &byte in chunk {
        counts[byte as usize] += 1;
    }

    let total = chunk.len() as f64;
    let inv_total = 1.0 / total;
    let log2_total = total.log2();

    // H = -Σ p_i * log2(p_i) = -Σ (c_i/n) * log2(c_i/n)
    //   = -Σ (c_i/n) * (log2(c_i) - log2(n))
    //   = -Σ (c_i/n) * log2(c_i) + Σ (c_i/n) * log2(n)
    //   = log2(n) - (1/n) * Σ c_i * log2(c_i)
    let mut sum_c_log_c = 0.0f64;
    for &count in &counts {
        if count > 0 {
            let c = count as f64;
            sum_c_log_c += c * c.log2();
        }
    }

    log2_total - inv_total * sum_c_log_c
}

/// Precompute entropy values for all chunks at a given interval.
/// Returns entropies for chunks starting at positions 0, interval, 2*interval, ...
pub fn precompute_chunk_entropies(data: &[u8], chunk_size: usize, interval: usize) -> Vec<f64> {
    if data.len() < chunk_size {
        return Vec::new();
    }

    let num_positions = (data.len() - chunk_size) / interval + 1;

    (0..num_positions)
        .into_par_iter()
        .map(|i| {
            let start = i * interval;
            let end = start + chunk_size;
            if end <= data.len() {
                chunk_entropy_fast(&data[start..end])
            } else {
                chunk_entropy_fast(&data[start..])
            }
        })
        .collect()
}

/// Highly optimized wavelet suspiciousness for exactly 8 entropy values.
/// Performs unrolled Haar wavelet transform and computes suspiciousness in one pass.
/// Zero heap allocations.
///
/// Uses orthonormal Haar wavelet (multiply by 1/√2 ≈ 0.7071) to match the paper's
/// energy distribution. See: Wojnowicz et al., "Wavelet Decomposition of Software
/// Entropy Reveals Symptoms of Malicious Code" (2016), Eq. 5.
#[inline(always)]
pub fn wavelet_suspiciousness_8(stream: &[f64; 8]) -> f64 {
    // Orthonormal Haar scaling factor
    const INV_SQRT2: f64 = std::f64::consts::FRAC_1_SQRT_2; // 1/√2 ≈ 0.7071

    // Unrolled Haar wavelet transform for 8 elements (3 levels)
    //
    // Level 2 (finest): 4 detail coefficients from pairs
    let d2_0 = (stream[0] - stream[1]) * INV_SQRT2;
    let d2_1 = (stream[2] - stream[3]) * INV_SQRT2;
    let d2_2 = (stream[4] - stream[5]) * INV_SQRT2;
    let d2_3 = (stream[6] - stream[7]) * INV_SQRT2;

    // Level 2 averages (for next level) - also use INV_SQRT2 for energy preservation
    let a2_0 = (stream[0] + stream[1]) * INV_SQRT2;
    let a2_1 = (stream[2] + stream[3]) * INV_SQRT2;
    let a2_2 = (stream[4] + stream[5]) * INV_SQRT2;
    let a2_3 = (stream[6] + stream[7]) * INV_SQRT2;

    // Level 1: 2 detail coefficients
    let d1_0 = (a2_0 - a2_1) * INV_SQRT2;
    let d1_1 = (a2_2 - a2_3) * INV_SQRT2;

    // Level 1 averages
    let a1_0 = (a2_0 + a2_1) * INV_SQRT2;
    let a1_1 = (a2_2 + a2_3) * INV_SQRT2;

    // Level 0 (coarsest): 1 detail coefficient
    let d0_0 = (a1_0 - a1_1) * INV_SQRT2;

    // Compute energies at each level
    let e2 = d2_0 * d2_0 + d2_1 * d2_1 + d2_2 * d2_2 + d2_3 * d2_3; // Fine (4 coeffs)
    let e1 = d1_0 * d1_0 + d1_1 * d1_1; // Mid (2 coeffs)
    let e0 = d0_0 * d0_0; // Coarse (1 coeff)

    let total_energy = e0 + e1 + e2;

    if total_energy < 1e-10 {
        return 0.5; // No variation, neutral suspiciousness
    }

    // Coarse energy ratio: levels 0 and 1 (first 2 of 3 levels)
    // Per paper: malware concentrates energy at coarse levels
    let coarse_energy = e0 + e1;
    let coarse_ratio = coarse_energy / total_energy;

    // Energy variance for additional suspiciousness signal
    let mean_energy = total_energy * (1.0 / 3.0);
    let var = (e0 - mean_energy) * (e0 - mean_energy)
        + (e1 - mean_energy) * (e1 - mean_energy)
        + (e2 - mean_energy) * (e2 - mean_energy);
    let std = (var * (1.0 / 3.0)).sqrt();
    let cv = std / mean_energy;

    // Combined suspiciousness
    // With orthonormal Haar, coarse ratios are relatively higher than unnormalized.
    // Adjusted baseline from 0.3 to 0.4 to account for this.
    // Maps: coarse_ratio 0.4 → 0, coarse_ratio 0.8 → 1
    let ratio_susp = ((coarse_ratio - 0.4) * 2.5).clamp(0.0, 1.0);
    let var_susp = (cv * 0.5).clamp(0.0, 1.0);

    ratio_susp * 0.7 + var_susp * 0.3
}

/// Optimized wavelet suspiciousness for 4 entropy values (2 levels).
/// Reserved for smaller window analysis.
#[allow(dead_code)]
#[inline(always)]
pub fn wavelet_suspiciousness_4(stream: &[f64; 4]) -> f64 {
    // Level 1 (finest): 2 detail coefficients
    let d1_0 = (stream[0] - stream[1]) * 0.5;
    let d1_1 = (stream[2] - stream[3]) * 0.5;

    // Level 1 averages
    let a1_0 = (stream[0] + stream[1]) * 0.5;
    let a1_1 = (stream[2] + stream[3]) * 0.5;

    // Level 0 (coarsest): 1 detail coefficient
    let d0_0 = (a1_0 - a1_1) * 0.5;

    // Energies
    let e1 = d1_0 * d1_0 + d1_1 * d1_1; // Fine
    let e0 = d0_0 * d0_0; // Coarse

    let total_energy = e0 + e1;

    if total_energy < 1e-10 {
        return 0.5;
    }

    // For 2 levels, coarse = level 0 only
    let coarse_ratio = e0 / total_energy;

    let mean_energy = total_energy * 0.5;
    let var = (e0 - mean_energy) * (e0 - mean_energy) + (e1 - mean_energy) * (e1 - mean_energy);
    let std = (var * 0.5).sqrt();
    let cv = std / mean_energy;

    let ratio_susp = ((coarse_ratio - 0.3) * 2.5).clamp(0.0, 1.0);
    let var_susp = (cv * 0.5).clamp(0.0, 1.0);

    ratio_susp * 0.7 + var_susp * 0.3
}

/// Optimized wavelet suspiciousness for 16 entropy values (4 levels).
/// Reserved for larger window analysis with more precision.
#[allow(dead_code)]
#[inline(always)]
pub fn wavelet_suspiciousness_16(stream: &[f64; 16]) -> f64 {
    // Level 3 (finest): 8 detail coefficients
    let d3: [f64; 8] = [
        (stream[0] - stream[1]) * 0.5,
        (stream[2] - stream[3]) * 0.5,
        (stream[4] - stream[5]) * 0.5,
        (stream[6] - stream[7]) * 0.5,
        (stream[8] - stream[9]) * 0.5,
        (stream[10] - stream[11]) * 0.5,
        (stream[12] - stream[13]) * 0.5,
        (stream[14] - stream[15]) * 0.5,
    ];

    // Level 3 averages
    let a3: [f64; 8] = [
        (stream[0] + stream[1]) * 0.5,
        (stream[2] + stream[3]) * 0.5,
        (stream[4] + stream[5]) * 0.5,
        (stream[6] + stream[7]) * 0.5,
        (stream[8] + stream[9]) * 0.5,
        (stream[10] + stream[11]) * 0.5,
        (stream[12] + stream[13]) * 0.5,
        (stream[14] + stream[15]) * 0.5,
    ];

    // Level 2: 4 detail coefficients
    let d2: [f64; 4] = [
        (a3[0] - a3[1]) * 0.5,
        (a3[2] - a3[3]) * 0.5,
        (a3[4] - a3[5]) * 0.5,
        (a3[6] - a3[7]) * 0.5,
    ];

    let a2: [f64; 4] = [
        (a3[0] + a3[1]) * 0.5,
        (a3[2] + a3[3]) * 0.5,
        (a3[4] + a3[5]) * 0.5,
        (a3[6] + a3[7]) * 0.5,
    ];

    // Level 1: 2 detail coefficients
    let d1_0 = (a2[0] - a2[1]) * 0.5;
    let d1_1 = (a2[2] - a2[3]) * 0.5;

    let a1_0 = (a2[0] + a2[1]) * 0.5;
    let a1_1 = (a2[2] + a2[3]) * 0.5;

    // Level 0 (coarsest): 1 detail coefficient
    let d0_0 = (a1_0 - a1_1) * 0.5;

    // Compute energies
    let e3: f64 = d3.iter().map(|x| x * x).sum();
    let e2: f64 = d2.iter().map(|x| x * x).sum();
    let e1 = d1_0 * d1_0 + d1_1 * d1_1;
    let e0 = d0_0 * d0_0;

    let total_energy = e0 + e1 + e2 + e3;

    if total_energy < 1e-10 {
        return 0.5;
    }

    // For 4 levels, coarse = levels 0 and 1 (first half)
    let coarse_energy = e0 + e1;
    let coarse_ratio = coarse_energy / total_energy;

    let mean_energy = total_energy * 0.25;
    let var = (e0 - mean_energy) * (e0 - mean_energy)
        + (e1 - mean_energy) * (e1 - mean_energy)
        + (e2 - mean_energy) * (e2 - mean_energy)
        + (e3 - mean_energy) * (e3 - mean_energy);
    let std = (var * 0.25).sqrt();
    let cv = std / mean_energy;

    let ratio_susp = ((coarse_ratio - 0.3) * 2.5).clamp(0.0, 1.0);
    let var_susp = (cv * 0.5).clamp(0.0, 1.0);

    ratio_susp * 0.7 + var_susp * 0.3
}

impl JSDAnalysis {
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

/// Format data as a hex dump string.
///
/// Format: `XXXXXXXX  XX XX XX XX XX XX XX XX  XX XX XX XX XX XX XX XX  |................|`
#[allow(dead_code)]
pub fn hex_dump(data: &[u8], start_offset: usize) -> String {
    const BYTES_PER_LINE: usize = 16;

    let mut output = String::with_capacity(data.len() * 4);
    let lines = (data.len() + BYTES_PER_LINE - 1) / BYTES_PER_LINE;

    for i in 0..lines {
        let offset = i * BYTES_PER_LINE;
        let end = (offset + BYTES_PER_LINE).min(data.len());
        let chunk = &data[offset..end];

        // Address
        let _ = write!(output, "{:08X}  ", start_offset + offset);

        // Hex bytes
        for (idx, &byte) in chunk.iter().enumerate() {
            let _ = write!(output, "{byte:02X} ");
            if idx == 7 {
                output.push(' ');
            }
        }

        // Padding for incomplete lines
        let remaining = BYTES_PER_LINE - chunk.len();
        if remaining > 0 {
            for _ in 0..remaining {
                output.push_str("   ");
            }
            if chunk.len() <= 8 {
                output.push(' ');
            }
        }

        output.push_str(" |");

        // ASCII representation
        for &byte in chunk {
            if byte >= 32 && byte <= 126 {
                output.push(byte as char);
            } else {
                output.push('.');
            }
        }

        output.push_str("|\n");
    }

    output
}

/// Identify file type via magic bytes.
pub fn identify_file_type(data: &[u8]) -> &'static str {
    if data.len() < 4 {
        return "Unknown Data";
    }

    // Helper macro for prefix checking
    macro_rules! has_prefix {
        ($bytes:expr) => {
            data.len() >= $bytes.len() && data.starts_with($bytes)
        };
    }

    // Check signatures in order of specificity
    if has_prefix!(&[0x4D, 0x5A]) {
        "Windows PE (EXE/DLL)"
    } else if has_prefix!(&[0x7F, 0x45, 0x4C, 0x46]) {
        "ELF Binary"
    } else if has_prefix!(&[0xFE, 0xED, 0xFA, 0xCE])
        || has_prefix!(&[0xCE, 0xFA, 0xED, 0xFE])
        || has_prefix!(&[0xCA, 0xFE, 0xBA, 0xBE])
        || has_prefix!(&[0xCF, 0xFA, 0xED, 0xFE])
    {
        "Mach-O Binary"
    } else if has_prefix!(&[0x25, 0x50, 0x44, 0x46]) {
        "PDF Document"
    } else if has_prefix!(&[0x50, 0x4B, 0x03, 0x04]) {
        "ZIP Archive / Office"
    } else if has_prefix!(&[0x89, 0x50, 0x4E, 0x47]) {
        "PNG Image"
    } else if has_prefix!(&[0xFF, 0xD8, 0xFF]) {
        "JPEG Image"
    } else if has_prefix!(&[0x47, 0x49, 0x46, 0x38]) {
        "GIF Image"
    } else if has_prefix!(&[0x52, 0x61, 0x72, 0x21]) {
        "RAR Archive"
    } else if has_prefix!(&[0x1F, 0x8B]) {
        "GZIP Archive"
    } else if has_prefix!(&[0x42, 0x5A, 0x68]) {
        "BZIP2 Archive"
    } else if has_prefix!(&[0x37, 0x7A, 0xBC, 0xAF]) {
        "7-Zip Archive"
    } else if has_prefix!(&[0x00, 0x00, 0x00])
        && data.len() > 4
        && (data[4] == 0x66 || data[4] == 0x6D)
    {
        "MP4/MOV Video"
    } else if has_prefix!(b"RIFF") {
        "RIFF Container (WAV/AVI)"
    } else if has_prefix!(b"SQLite") {
        "SQLite Database"
    } else {
        "Unknown Binary"
    }
}

/// Extract the first printable ASCII string (length > 4) from data.
#[allow(dead_code)]
pub fn extract_ascii(data: &[u8]) -> Option<String> {
    let mut current = String::new();

    for &byte in data {
        if byte >= 32 && byte <= 126 {
            current.push(byte as char);
        } else {
            if current.len() > 4 {
                return Some(current);
            }
            current.clear();
        }
    }

    if current.len() > 4 {
        Some(current)
    } else {
        None
    }
}

/// Byte analysis results for color mapping.
#[derive(Debug, Clone, Copy)]
pub struct ByteAnalysis {
    /// Proportion of printable ASCII text (0.0 - 1.0)
    pub text_ratio: f32,
    /// Proportion of high-bit bytes (0.0 - 1.0)
    pub high_ratio: f32,
    /// Proportion of null bytes (0.0 - 1.0)
    pub null_ratio: f32,
    /// Normalized variation/entropy estimate (0.0 - 1.0)
    pub variation: f32,
}

impl ByteAnalysis {
    /// Analyze a chunk of bytes for forensic color mapping.
    pub fn analyze(data: &[u8]) -> Self {
        if data.is_empty() {
            return Self {
                text_ratio: 0.0,
                high_ratio: 0.0,
                null_ratio: 0.0,
                variation: 0.0,
            };
        }

        let mut text_chars = 0u32;
        let mut high_bits = 0u32;
        let mut nulls = 0u32;
        let mut variation = 0u32;

        let mut prev_byte: Option<u8> = None;

        for &byte in data {
            if (32..=126).contains(&byte) {
                text_chars += 1;
            }
            if byte > 127 {
                high_bits += 1;
            }
            if byte == 0 {
                nulls += 1;
            }
            if let Some(prev) = prev_byte {
                let diff = (byte as i16 - prev as i16).unsigned_abs() as u32;
                variation += diff;
            }
            prev_byte = Some(byte);
        }

        let count = data.len() as f32;

        Self {
            text_ratio: text_chars as f32 / count,
            high_ratio: high_bits as f32 / count,
            null_ratio: nulls as f32 / count,
            variation: (variation as f32 / count) / 128.0,
        }
    }

    /// Map analysis to RGB color using forensic color scheme.
    ///
    /// Color legend:
    /// - Blue: Nulls / padding / zeroes
    /// - Cyan: ASCII text
    /// - Green: Code / structured data
    /// - Red: High entropy / encrypted data
    pub fn to_color(&self) -> [u8; 3] {
        // Padding / Zeroes -> Deep Blue
        if self.null_ratio > 0.9 {
            let intensity = (0.2 + 0.3 * self.null_ratio).min(1.0);
            return [0, 0, (intensity * 255.0) as u8];
        }

        // High Entropy / Encryption -> Red
        if self.variation > 0.5 && self.high_ratio > 0.25 {
            return [255, 0, 0];
        }

        // ASCII Text -> Cyan
        if self.text_ratio > 0.85 {
            let intensity = (0.8 * self.variation + 0.2).min(1.0);
            let val = (intensity * 255.0) as u8;
            return [0, val, val];
        }

        // Code / Machine Instructions -> Green
        let intensity = (0.5 + 0.5 * self.variation).min(1.0);
        [0, (intensity * 255.0) as u8, 0]
    }
}

/// Format byte count as human-readable string.
pub fn format_bytes(bytes: u64) -> String {
    const KB: u64 = 1024;
    const MB: u64 = KB * 1024;
    const GB: u64 = MB * 1024;

    if bytes >= GB {
        format!("{:.1} GB", bytes as f64 / GB as f64)
    } else if bytes >= MB {
        format!("{:.1} MB", bytes as f64 / MB as f64)
    } else if bytes >= KB {
        format!("{:.1} KB", bytes as f64 / KB as f64)
    } else {
        format!("{bytes} B")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_entropy_uniform() {
        let data = vec![0u8; 100];
        assert!(calculate_entropy(&data) < 0.01);
    }

    #[test]
    fn test_entropy_random() {
        let data: Vec<u8> = (0..=255).collect();
        let entropy = calculate_entropy(&data);
        assert!(entropy > 7.9 && entropy <= 8.0);
    }

    #[test]
    fn test_file_type_detection() {
        assert_eq!(
            identify_file_type(&[0x4D, 0x5A, 0x00, 0x00]),
            "Windows PE (EXE/DLL)"
        );
        assert_eq!(identify_file_type(&[0x7F, 0x45, 0x4C, 0x46]), "ELF Binary");
        assert_eq!(identify_file_type(&[0x89, 0x50, 0x4E, 0x47]), "PNG Image");
    }

    #[test]
    fn test_extract_ascii() {
        let data = b"\x00\x00hello world\x00\x00";
        assert_eq!(extract_ascii(data), Some("hello world".to_string()));
    }

    #[test]
    fn test_format_bytes() {
        assert_eq!(format_bytes(500), "500 B");
        assert_eq!(format_bytes(1024), "1.0 KB");
        assert_eq!(format_bytes(1024 * 1024), "1.0 MB");
    }

    #[test]
    fn test_kolmogorov_complexity_uniform() {
        // Highly compressible data (all zeros) should have low complexity
        let data = vec![0u8; 1000];
        let complexity = calculate_kolmogorov_complexity(&data);
        assert!(
            complexity < 0.1,
            "Uniform data should have low complexity: {complexity}"
        );
    }

    #[test]
    fn test_kolmogorov_complexity_random() {
        // Random-like data should have higher complexity than uniform data
        // Use a more chaotic sequence (LCG with different parameters)
        let mut x: u64 = 12345;
        let data: Vec<u8> = (0..1000)
            .map(|_| {
                x = x.wrapping_mul(1103515245).wrapping_add(12345);
                ((x >> 16) & 0xFF) as u8
            })
            .collect();
        let complexity = calculate_kolmogorov_complexity(&data);
        // Even pseudo-random may compress somewhat; just check it's higher than repetitive
        assert!(
            complexity > 0.2,
            "Random-like data should have higher complexity: {complexity}"
        );
    }

    #[test]
    fn test_kolmogorov_complexity_repetitive() {
        // Repetitive pattern should have low complexity
        let pattern = b"ABCD";
        let data: Vec<u8> = pattern.iter().cycle().take(1000).copied().collect();
        let complexity = calculate_kolmogorov_complexity(&data);
        assert!(
            complexity < 0.2,
            "Repetitive data should have low complexity: {complexity}"
        );
    }

    #[test]
    fn test_coarse_grain_basic() {
        // Test coarse-graining at scale 2
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let mut coarse = vec![0.0; 10];
        let len = super::coarse_grain_into(&data, 2, 1, &mut coarse);
        // Expected: [(1+2)/2, (3+4)/2, (5+6)/2, (7+8)/2] = [1.5, 3.5, 5.5, 7.5]
        assert_eq!(len, 4);
        assert!((coarse[0] - 1.5).abs() < 0.01);
        assert!((coarse[1] - 3.5).abs() < 0.01);
        assert!((coarse[2] - 5.5).abs() < 0.01);
        assert!((coarse[3] - 7.5).abs() < 0.01);
    }

    #[test]
    fn test_coarse_grain_offset() {
        // Test coarse-graining at scale 2 with offset 2
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let mut coarse = vec![0.0; 10];
        let len = super::coarse_grain_into(&data, 2, 2, &mut coarse);
        // With offset 2: start at index 1, [(2+3)/2, (4+5)/2, (6+7)/2] = [2.5, 4.5, 6.5]
        assert_eq!(len, 3);
        assert!((coarse[0] - 2.5).abs() < 0.01);
        assert!((coarse[1] - 4.5).abs() < 0.01);
        assert!((coarse[2] - 6.5).abs() < 0.01);
    }

    #[test]
    fn test_rcmse_uniform_data() {
        // Uniform data should have very low entropy (highly predictable)
        let data = vec![128u8; 500];
        let analysis = super::calculate_rcmse(&data, 8);
        // Uniform data has zero variance, r=0, so we get Unknown classification
        assert_eq!(analysis.classification, super::RCMSEClassification::Unknown);
    }

    #[test]
    fn test_rcmse_random_data() {
        // Pseudo-random data should be classified as Random
        let mut x: u64 = 42;
        let data: Vec<u8> = (0..1000)
            .map(|_| {
                x = x.wrapping_mul(1103515245).wrapping_add(12345);
                ((x >> 16) & 0xFF) as u8
            })
            .collect();
        let analysis = super::calculate_rcmse(&data, 10);

        // Random data should have negative slope (entropy decreases with scale)
        assert!(
            analysis.slope < 0.0,
            "Random data should have negative slope: {}",
            analysis.slope
        );
    }

    #[test]
    fn test_rcmse_quick() {
        // Test the quick RCMSE function returns valid range
        let data: Vec<u8> = (0..500).map(|i| (i % 256) as u8).collect();
        let value = super::calculate_rcmse_quick(&data);
        assert!(
            value >= 0.0 && value <= 1.0,
            "RCMSE quick value out of range: {}",
            value
        );
    }

    #[test]
    fn test_rcmse_analysis_to_color() {
        // Test that color mapping produces non-zero RGB values for various classifications
        let classifications = [
            super::RCMSEClassification::Random,
            super::RCMSEClassification::Structured,
            super::RCMSEClassification::Chaotic,
            super::RCMSEClassification::Unknown,
        ];

        for classification in classifications {
            let analysis = super::RCMSEAnalysis {
                entropy_by_scale: vec![2.0, 1.9, 1.8, 1.7],
                slope: -0.1,
                complexity_index: 0.5,
                mean_entropy: 1.85,
                classification,
            };
            let color = analysis.to_color();
            // Verify we get non-zero colors (not all black)
            assert!(
                color[0] > 0 || color[1] > 0 || color[2] > 0,
                "Color should not be all zeros for {:?}",
                classification
            );
        }
    }

    // =============================================================================
    // Wavelet Entropy Tests
    // =============================================================================

    #[test]
    fn test_compute_entropy_stream() {
        // Test entropy stream computation
        let data: Vec<u8> = (0..512).map(|i| (i % 256) as u8).collect();
        let stream = super::compute_entropy_stream(&data);
        // 512 bytes / 256 chunk size = 2 chunks
        assert_eq!(stream.len(), 2);
        // Each chunk has all 256 values once, so max entropy
        assert!(stream[0] > 7.9, "Expected high entropy: {}", stream[0]);
    }

    #[test]
    fn test_haar_wavelet_transform_basic() {
        // Test with a simple signal of length 8
        let signal = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let coefficients = super::haar_wavelet_transform(&signal);

        // 8 = 2^3, so we should have 3 levels
        assert_eq!(coefficients.len(), 3);

        // Level 0 (coarsest): 1 coefficient
        assert_eq!(coefficients[0].len(), 1);
        // Level 1: 2 coefficients
        assert_eq!(coefficients[1].len(), 2);
        // Level 2 (finest): 4 coefficients
        assert_eq!(coefficients[2].len(), 4);
    }

    #[test]
    fn test_haar_wavelet_transform_constant() {
        // Constant signal should have zero detail coefficients
        let signal = vec![5.0; 8];
        let coefficients = super::haar_wavelet_transform(&signal);

        // All detail coefficients should be (near) zero
        for level in &coefficients {
            for &coeff in level {
                assert!(
                    coeff.abs() < 1e-10,
                    "Constant signal should have zero wavelet coefficients"
                );
            }
        }
    }

    #[test]
    fn test_wavelet_energy_spectrum() {
        // Test energy calculation
        let coefficients = vec![vec![2.0], vec![1.0, -1.0], vec![0.5, 0.5, -0.5, -0.5]];
        let energy = super::wavelet_energy_spectrum(&coefficients);

        assert_eq!(energy.len(), 3);
        assert!((energy[0] - 4.0).abs() < 1e-10); // 2^2 = 4
        assert!((energy[1] - 2.0).abs() < 1e-10); // 1^2 + 1^2 = 2
        assert!((energy[2] - 1.0).abs() < 1e-10); // 4 * 0.25 = 1
    }

    #[test]
    fn test_wavelet_analysis_uniform_data() {
        // Uniform data should have low suspiciousness
        let data = vec![128u8; 1024];
        let stream = super::compute_entropy_stream(&data);
        let analysis = super::WaveletAnalysis::from_entropy_stream(&stream);

        // Uniform data = constant entropy = zero wavelet coefficients
        // Should result in middle-range suspiciousness
        assert!(
            analysis.total_energy < 0.1,
            "Uniform data should have near-zero wavelet energy"
        );
    }

    #[test]
    fn test_wavelet_analysis_to_color() {
        // Test that color mapping produces valid RGB values
        let stream = vec![1.0, 5.0, 7.0, 2.0, 6.0, 3.0, 4.0, 5.0];
        let analysis = super::WaveletAnalysis::from_entropy_stream(&stream);
        let color = analysis.to_color();

        // Verify valid RGB values
        assert!(color[0] <= 255);
        assert!(color[1] <= 255);
        assert!(color[2] <= 255);
        // Should have some color (not all black)
        assert!(
            color[0] > 0 || color[1] > 0 || color[2] > 0,
            "Color should not be all zeros"
        );
    }

    #[test]
    fn test_calculate_wavelet_suspiciousness() {
        // Test quick suspiciousness calculation
        let data: Vec<u8> = (0..2048).map(|i| (i % 256) as u8).collect();
        let suspiciousness = super::calculate_wavelet_suspiciousness(&data);

        // Should be in valid range
        assert!(
            suspiciousness >= 0.0 && suspiciousness <= 1.0,
            "Suspiciousness out of range: {}",
            suspiciousness
        );
    }

    #[test]
    fn test_optimized_wavelet_matches_original() {
        // Verify that wavelet_suspiciousness_8 produces same results as WaveletAnalysis
        let test_streams: [[f64; 8]; 4] = [
            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], // Linear ramp
            [5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0], // Constant
            [1.0, 7.0, 2.0, 6.0, 3.0, 5.0, 4.0, 4.0], // Varying
            [7.5, 7.5, 7.5, 7.5, 1.0, 1.0, 1.0, 1.0], // Step (suspicious pattern)
        ];

        for stream in &test_streams {
            let original = super::WaveletAnalysis::from_entropy_stream(stream);
            let optimized = super::wavelet_suspiciousness_8(stream);

            let diff = (original.suspiciousness - optimized).abs();
            assert!(
                diff < 1e-10,
                "Mismatch: original={}, optimized={}, diff={}",
                original.suspiciousness,
                optimized,
                diff
            );
        }
    }

    #[test]
    fn test_precompute_chunk_entropies() {
        // Test that precompute_chunk_entropies produces same results as individual calls
        let data: Vec<u8> = (0..1024).map(|i| (i % 256) as u8).collect();
        let chunk_size = 256;
        let interval = 64;

        let precomputed = super::precompute_chunk_entropies(&data, chunk_size, interval);

        // Verify each precomputed value matches direct calculation
        for (i, &precomputed_entropy) in precomputed.iter().enumerate() {
            let start = i * interval;
            let end = (start + chunk_size).min(data.len());
            let direct_entropy = super::chunk_entropy(&data[start..end]);

            let diff = (precomputed_entropy - direct_entropy).abs();
            assert!(
                diff < 1e-10,
                "Chunk {} mismatch: precomputed={}, direct={}, diff={}",
                i,
                precomputed_entropy,
                direct_entropy,
                diff
            );
        }
    }

    #[test]
    fn test_chunk_entropy_fast_matches_original() {
        // Verify chunk_entropy_fast produces same results as chunk_entropy
        let test_chunks: [&[u8]; 4] = [
            &[0u8; 256],                                                 // All zeros
            &(0..=255).collect::<Vec<u8>>(),                             // All values
            &vec![0xAB; 256],                                            // Single value
            &(0..256).map(|i| (i * 7 % 256) as u8).collect::<Vec<u8>>(), // Pseudo-random
        ];

        for chunk in test_chunks {
            let original = super::chunk_entropy(chunk);
            let fast = super::chunk_entropy_fast(chunk);

            let diff = (original - fast).abs();
            assert!(
                diff < 1e-10,
                "Entropy mismatch: original={}, fast={}, diff={}",
                original,
                fast,
                diff
            );
        }
    }
}
