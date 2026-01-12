//! Pixel generation functions for binary visualizations.
//!
//! Each generator produces a Vec<Color32> for rendering visualization textures.

use eframe::egui::{Color32, Vec2};
use rayon::prelude::*;

use crate::analysis::{
    calculate_jsd, ByteAnalysis, JSDAnalysis, KolmogorovAnalysis, ANALYSIS_WINDOW,
};
use crate::hilbert::{xy2d, HilbertBuffer};

// =============================================================================
// Color Utilities
// =============================================================================

/// Convert HSV to RGB Color32.
///
/// - hue: 0-360 degrees
/// - saturation: 0-1
/// - value: 0-1
pub fn hsv_to_rgb(hue: f64, saturation: f64, value: f64) -> Color32 {
    let h = (hue % 360.0) / 60.0;
    let c = value * saturation;
    let x = c * (1.0 - (h % 2.0 - 1.0).abs());
    let m = value - c;

    let (r, g, b) = if h < 1.0 {
        (c, x, 0.0)
    } else if h < 2.0 {
        (x, c, 0.0)
    } else if h < 3.0 {
        (0.0, c, x)
    } else if h < 4.0 {
        (0.0, x, c)
    } else if h < 5.0 {
        (x, 0.0, c)
    } else {
        (c, 0.0, x)
    };

    Color32::from_rgb(
        ((r + m) * 255.0) as u8,
        ((g + m) * 255.0) as u8,
        ((b + m) * 255.0) as u8,
    )
}

/// Map similarity value (0-1) to color for recurrence plots.
/// Uses "inferno" colormap style for better perceptual uniformity.
///
/// - 0 (dissimilar): Black/dark purple
/// - 0.5 (moderate): Red/orange
/// - 1.0 (identical): Bright yellow/white
/// - Diagonal elements get special highlighting
pub fn similarity_to_color(similarity: f64, is_diagonal: bool) -> Color32 {
    // Apply gamma curve to enhance contrast in the mid-range
    let t = similarity.clamp(0.0, 1.0).powf(0.4); // Gamma < 1 brightens dark areas

    if is_diagonal {
        // Diagonal elements (self-comparison) - bright cyan/white
        return Color32::from_rgb((180.0 + t * 75.0) as u8, (220.0 + t * 35.0) as u8, 255);
    }

    // Inferno-inspired colormap: black -> purple -> red -> orange -> yellow -> white
    let (r, g, b) = if t < 0.2 {
        // Black to dark purple
        let s = t / 0.2;
        (s * 0.25, 0.0, s * 0.35)
    } else if t < 0.4 {
        // Dark purple to red-purple
        let s = (t - 0.2) / 0.2;
        (0.25 + s * 0.45, 0.0, 0.35 + s * 0.1)
    } else if t < 0.6 {
        // Red-purple to orange-red
        let s = (t - 0.4) / 0.2;
        (0.7 + s * 0.25, s * 0.25, 0.45 - s * 0.35)
    } else if t < 0.8 {
        // Orange-red to orange-yellow
        let s = (t - 0.6) / 0.2;
        (0.95 + s * 0.05, 0.25 + s * 0.45, 0.1 + s * 0.1)
    } else {
        // Orange-yellow to bright yellow-white
        let s = (t - 0.8) / 0.2;
        (1.0, 0.7 + s * 0.3, 0.2 + s * 0.7)
    };

    Color32::from_rgb((r * 255.0) as u8, (g * 255.0) as u8, (b * 255.0) as u8)
}

// =============================================================================
// Sample Interval Constants
// =============================================================================

/// Interval for sampling Kolmogorov complexity (every N bytes).
const COMPLEXITY_SAMPLE_INTERVAL: usize = 64;

/// Interval for sampling RCMSE (every N bytes).
const RCMSE_SAMPLE_INTERVAL: usize = 64;

/// Interval for sampling wavelet suspiciousness (every N bytes).
const WAVELET_SAMPLE_INTERVAL: usize = 64;

// =============================================================================
// Lookup Functions
// =============================================================================

/// Look up precomputed complexity value for a file offset.
fn lookup_complexity(complexity_map: &[f32], offset: usize) -> f32 {
    if complexity_map.is_empty() {
        return 0.0;
    }
    let index = offset / COMPLEXITY_SAMPLE_INTERVAL;
    complexity_map.get(index).copied().unwrap_or(0.0)
}

/// Look up precomputed RCMSE value for a file offset.
fn lookup_rcmse(rcmse_map: &[f32], offset: usize) -> f32 {
    if rcmse_map.is_empty() {
        return 0.5;
    }
    let index = offset / RCMSE_SAMPLE_INTERVAL;
    rcmse_map.get(index).copied().unwrap_or(0.5)
}

/// Look up precomputed wavelet suspiciousness value for a file offset.
fn lookup_wavelet(wavelet_map: &[f32], offset: usize) -> f32 {
    if wavelet_map.is_empty() {
        return 0.5;
    }
    let index = offset / WAVELET_SAMPLE_INTERVAL;
    wavelet_map.get(index).copied().unwrap_or(0.5)
}

// =============================================================================
// Hilbert Curve Visualization
// =============================================================================

/// Generate pixels using Hilbert curve mapping with entropy coloring.
/// Used for small files that can be computed immediately.
pub fn generate_hilbert_pixels(
    data: &[u8],
    dimension: u64,
    file_size: u64,
    tex_size: usize,
    world_min: Vec2,
    scale_x: f32,
    scale_y: f32,
) -> Vec<Color32> {
    (0..tex_size * tex_size)
        .into_par_iter()
        .map(|idx| {
            let tex_y = idx / tex_size;
            let tex_x = idx % tex_size;

            let world_x = (world_min.x + tex_x as f32 * scale_x) as u64;
            let world_y = (world_min.y + tex_y as f32 * scale_y) as u64;

            if world_x >= dimension || world_y >= dimension {
                return Color32::from_rgb(13, 13, 13);
            }

            let d = xy2d(dimension, world_x, world_y);

            if d >= file_size {
                return Color32::from_rgb(13, 13, 13);
            }

            let start = d as usize;
            let end = (start + ANALYSIS_WINDOW).min(data.len());
            let chunk = &data[start..end];

            let analysis = ByteAnalysis::analyze(chunk);
            let [r, g, b] = analysis.to_color();
            Color32::from_rgb(r, g, b)
        })
        .collect()
}

/// Generate pixels using progressive Hilbert computation buffer.
/// Shows computed regions with colors, uncomputed regions with animated pulse.
pub fn generate_hilbert_pixels_progressive(
    buffer: &HilbertBuffer,
    dimension: u64,
    file_size: u64,
    tex_size: usize,
    world_min: Vec2,
    scale_x: f32,
    scale_y: f32,
    time_seconds: f64,
) -> Vec<Color32> {
    use crate::hilbert::{entropy_to_color, entropy_to_color_preview, placeholder_pulse_color};

    (0..tex_size * tex_size)
        .into_par_iter()
        .map(|idx| {
            let tex_y = idx / tex_size;
            let tex_x = idx % tex_size;

            let world_x = (world_min.x + tex_x as f32 * scale_x) as u64;
            let world_y = (world_min.y + tex_y as f32 * scale_y) as u64;

            if world_x >= dimension || world_y >= dimension {
                return Color32::from_rgb(13, 13, 13);
            }

            let d = xy2d(dimension, world_x, world_y);

            if d >= file_size {
                return Color32::from_rgb(13, 13, 13);
            }

            // Look up from progressive buffer
            match buffer.lookup(d) {
                Some((entropy, is_fine)) => {
                    let (r, g, b) = if is_fine {
                        entropy_to_color(entropy)
                    } else {
                        entropy_to_color_preview(entropy)
                    };
                    Color32::from_rgb(r, g, b)
                }
                None => {
                    // Not yet computed - show animated pulse
                    let (r, g, b) = placeholder_pulse_color(time_seconds);
                    Color32::from_rgb(r, g, b)
                }
            }
        })
        .collect()
}

/// Generate a simple animated placeholder while data is being computed.
/// Uses NO parallel iteration to avoid competing with background computation.
pub fn generate_computing_placeholder(tex_size: usize, time_seconds: f64) -> Vec<Color32> {
    let total = tex_size * tex_size;
    let mut pixels = Vec::with_capacity(total);

    // Gentle animated gradient to indicate "working"
    let pulse = ((time_seconds * 2.0 * std::f64::consts::PI).sin() * 0.5 + 0.5) as f32;
    let base_color = 25.0 + pulse * 10.0;

    for idx in 0..total {
        let y = idx / tex_size;
        let x = idx % tex_size;

        // Subtle diagonal gradient pattern
        let gradient = ((x + y) as f32 / (tex_size * 2) as f32) * 15.0;
        let v = (base_color + gradient) as u8;

        // Slight blue tint to indicate "computing"
        pixels.push(Color32::from_rgb(v, v, v.saturating_add(12)));
    }

    pixels
}

// =============================================================================
// Similarity Matrix Visualization
// =============================================================================

/// Generate Structural Similarity Matrix (Recurrence Plot) with viewport support.
///
/// Each pixel (i, j) shows the similarity between byte windows at
/// positions i and j in the file. Supports pan/zoom for detailed exploration.
///
/// - Diagonal lines: Repeating patterns/sequences
/// - Vertical/horizontal lines: Laminar states (unchanged regions)
/// - Checkerboard patterns: Periodic structures
pub fn generate_similarity_matrix_pixels(
    data: &[u8],
    file_size: u64,
    tex_size: usize,
    world_min: Vec2,
    scale_x: f32,
    scale_y: f32,
    dimension: u64,
) -> Vec<Color32> {
    let window_size: usize = 32; // Comparison window
    let dim = dimension as f64;

    // Precompute histograms for file positions we'll need
    // Map world coordinates to file positions
    let get_histogram = |file_pos: usize| -> [u32; 256] {
        let mut hist = [0u32; 256];
        let end = (file_pos + window_size).min(data.len());
        if file_pos < data.len() {
            for &byte in &data[file_pos..end] {
                hist[byte as usize] += 1;
            }
        }
        hist
    };

    (0..tex_size * tex_size)
        .into_par_iter()
        .map(|idx| {
            let tex_y = idx / tex_size;
            let tex_x = idx % tex_size;

            // Map texture coordinates to world coordinates
            let world_x = world_min.x + tex_x as f32 * scale_x;
            let world_y = world_min.y + tex_y as f32 * scale_y;

            if world_x < 0.0 || world_y < 0.0 || world_x >= dim as f32 || world_y >= dim as f32 {
                return Color32::from_rgb(10, 10, 15);
            }

            // Map world coordinates to file positions
            let pos_x = ((world_x as f64 / dim) * file_size as f64) as usize;
            let pos_y = ((world_y as f64 / dim) * file_size as f64) as usize;

            let hist_x = get_histogram(pos_x);
            let hist_y = get_histogram(pos_y);

            // Compute chi-squared distance (more sensitive to differences than Jaccard)
            let mut chi_sq = 0.0f64;
            let mut total_x = 0u32;
            let mut total_y = 0u32;

            for i in 0..256 {
                total_x += hist_x[i];
                total_y += hist_y[i];
            }

            if total_x > 0 && total_y > 0 {
                for i in 0..256 {
                    let px = hist_x[i] as f64 / total_x as f64;
                    let py = hist_y[i] as f64 / total_y as f64;
                    let sum = px + py;
                    if sum > 0.0 {
                        chi_sq += (px - py).powi(2) / sum;
                    }
                }
            }

            // Convert distance to similarity (chi_sq ranges from 0 to 2)
            // Apply sqrt for better spread, then invert
            let similarity = 1.0 - (chi_sq / 2.0).sqrt();

            let is_diagonal = (pos_x as i64 - pos_y as i64).abs() < window_size as i64;
            similarity_to_color(similarity, is_diagonal)
        })
        .collect()
}

// =============================================================================
// Digraph Visualization
// =============================================================================

/// Generate Byte Digraph with viewport support.
///
/// Shows byte pair transition frequencies. X=source byte, Y=destination byte.
/// The digraph is mapped to a 256x256 logical space that can be zoomed/panned.
pub fn generate_digraph_pixels(
    data: &[u8],
    file_size: u64,
    tex_size: usize,
    world_min: Vec2,
    scale_x: f32,
    scale_y: f32,
) -> Vec<Color32> {
    // Build the digraph: count transitions from byte A to byte B
    let mut digraph = vec![0u64; 256 * 256];
    let mut max_count = 1u64;

    if file_size >= 2 {
        for window in data.windows(2) {
            let from = window[0] as usize;
            let to = window[1] as usize;
            let idx = from * 256 + to;
            digraph[idx] += 1;
            max_count = max_count.max(digraph[idx]);
        }
    }

    let sqrt_max = (max_count as f64).sqrt();

    // Generate texture with viewport support
    // World space is 256x256 (byte values), scale to dimension for consistency
    (0..tex_size * tex_size)
        .into_par_iter()
        .map(|idx| {
            let tex_y = idx / tex_size;
            let tex_x = idx % tex_size;

            // Map texture coordinates to world coordinates
            let world_x = world_min.x + tex_x as f32 * scale_x;
            let world_y = world_min.y + tex_y as f32 * scale_y;

            // Scale world coordinates (which are in dimension-space) to 0-255
            // For digraph, we want to show 256x256 in the center of the view
            let dg_x = world_x as i32;
            let dg_y = world_y as i32;

            if dg_x < 0 || dg_x >= 256 || dg_y < 0 || dg_y >= 256 {
                return Color32::from_rgb(8, 8, 12);
            }

            let count = digraph[dg_y as usize * 256 + dg_x as usize];

            if count == 0 {
                Color32::from_rgb(8, 8, 12)
            } else {
                // Use sqrt for better contrast distribution than log
                let intensity = ((count as f64).sqrt() / sqrt_max).powf(0.6);

                // Color by byte values with full spectrum
                let hue = ((dg_x + dg_y) as f64 / 512.0) * 300.0; // More hue range
                let sat = 0.85 - intensity * 0.2; // High count = slightly less saturated
                let val = 0.15 + intensity * 0.85; // Wider brightness range

                hsv_to_rgb(hue, sat, val)
            }
        })
        .collect()
}

// =============================================================================
// Byte Phase Space Visualization
// =============================================================================

/// Generate Byte Phase Space with viewport support.
///
/// Plots byte[i] vs byte[i+1] for all sequential bytes, colored by file position.
/// This creates a phase space trajectory showing the file's "attractor".
pub fn generate_byte_phase_space_pixels(
    data: &[u8],
    file_size: u64,
    tex_size: usize,
    world_min: Vec2,
    scale_x: f32,
    scale_y: f32,
) -> Vec<Color32> {
    if file_size < 2 {
        return vec![Color32::from_rgb(10, 10, 15); tex_size * tex_size];
    }

    // Build phase space: for each pixel, store the latest file position that hits it
    // and total count for alpha/intensity
    let mut phase_space = vec![(0u64, 0u64); 256 * 256]; // (last_position, count)

    for (i, window) in data.windows(2).enumerate() {
        let x = window[0] as usize;
        let y = window[1] as usize;
        let idx = y * 256 + x;
        phase_space[idx] = (i as u64, phase_space[idx].1 + 1);
    }

    let max_count = phase_space
        .iter()
        .map(|(_, c)| *c)
        .max()
        .unwrap_or(1)
        .max(1);
    let file_size_f = file_size as f64;
    let sqrt_max = (max_count as f64).sqrt();

    // Generate texture with viewport support
    (0..tex_size * tex_size)
        .into_par_iter()
        .map(|idx| {
            let tex_y = idx / tex_size;
            let tex_x = idx % tex_size;

            // Map texture coordinates to world coordinates
            let world_x = world_min.x + tex_x as f32 * scale_x;
            let world_y = world_min.y + tex_y as f32 * scale_y;

            let ps_x = world_x as i32;
            let ps_y = world_y as i32;

            if ps_x < 0 || ps_x >= 256 || ps_y < 0 || ps_y >= 256 {
                return Color32::from_rgb(10, 10, 15);
            }

            let (last_pos, count) = phase_space[ps_y as usize * 256 + ps_x as usize];

            if count == 0 {
                Color32::from_rgb(8, 8, 12)
            } else {
                // Use sqrt for better contrast distribution
                let intensity = ((count as f64).sqrt() / sqrt_max).powf(0.5);

                // Color by byte coordinates for spatial structure
                let coord_hue = ((ps_x + ps_y) as f64 / 512.0) * 360.0;

                // Blend position-based hue with coordinate-based hue
                let position_ratio = last_pos as f64 / file_size_f;
                let pos_hue = position_ratio * 120.0; // 0 (red) -> 120 (green)

                let hue = (coord_hue * 0.7 + pos_hue * 0.3) % 360.0;
                let sat = 0.75 + intensity * 0.2;
                let val = 0.1 + intensity * 0.9;

                hsv_to_rgb(hue, sat, val)
            }
        })
        .collect()
}

// =============================================================================
// Kolmogorov Complexity Visualization
// =============================================================================

/// Generate Kolmogorov Complexity visualization using Hilbert curve mapping.
///
/// Uses precomputed complexity values for fast rendering.
/// Each pixel shows the compression ratio of a window starting at that file offset.
/// This reveals algorithmic complexity patterns - regions that are simple/repetitive
/// vs. complex/random.
///
/// Color legend (viridis-inspired):
/// - Purple/blue: Low complexity (highly compressible, simple patterns)
/// - Teal/green: Medium complexity (structured data)
/// - Yellow/orange: High complexity (compressed/encrypted data)
/// - Red/pink: Maximum complexity (truly random data)
pub fn generate_kolmogorov_pixels(
    complexity_map: &[f32],
    dimension: u64,
    file_size: u64,
    tex_size: usize,
    world_min: Vec2,
    scale_x: f32,
    scale_y: f32,
) -> Vec<Color32> {
    (0..tex_size * tex_size)
        .into_par_iter()
        .map(|idx| {
            let tex_y = idx / tex_size;
            let tex_x = idx % tex_size;

            let world_x = (world_min.x + tex_x as f32 * scale_x) as u64;
            let world_y = (world_min.y + tex_y as f32 * scale_y) as u64;

            if world_x >= dimension || world_y >= dimension {
                return Color32::from_rgb(13, 13, 13);
            }

            let d = xy2d(dimension, world_x, world_y);

            if d >= file_size {
                return Color32::from_rgb(13, 13, 13);
            }

            // Use precomputed complexity value
            let complexity = lookup_complexity(complexity_map, d as usize);
            let analysis = KolmogorovAnalysis {
                complexity: complexity as f64,
            };
            let [r, g, b] = analysis.to_color();
            Color32::from_rgb(r, g, b)
        })
        .collect()
}

// =============================================================================
// Jensen-Shannon Divergence Visualization
// =============================================================================

/// Generate Jensen-Shannon Divergence visualization using Hilbert curve mapping.
///
/// Shows how different each region's byte distribution is from the overall file.
/// Highlights anomalous regions with unusual byte distributions.
///
/// Color legend:
/// - Dark blue/purple: Very similar to file average (normal regions)
/// - Cyan/teal: Slightly different distribution
/// - Green/yellow: Moderately different (interesting regions)
/// - Orange/red: Very different (anomalous regions - headers, compressed sections, etc.)
pub fn generate_jsd_pixels(
    data: &[u8],
    reference_distribution: &[f64; 256],
    dimension: u64,
    file_size: u64,
    tex_size: usize,
    world_min: Vec2,
    scale_x: f32,
    scale_y: f32,
) -> Vec<Color32> {
    (0..tex_size * tex_size)
        .into_par_iter()
        .map(|idx| {
            let tex_y = idx / tex_size;
            let tex_x = idx % tex_size;

            let world_x = (world_min.x + tex_x as f32 * scale_x) as u64;
            let world_y = (world_min.y + tex_y as f32 * scale_y) as u64;

            if world_x >= dimension || world_y >= dimension {
                return Color32::from_rgb(13, 13, 13);
            }

            let d = xy2d(dimension, world_x, world_y);

            if d >= file_size {
                return Color32::from_rgb(13, 13, 13);
            }

            let start = d as usize;
            let end = (start + ANALYSIS_WINDOW).min(data.len());
            let chunk = &data[start..end];

            // Calculate JSD against reference distribution
            let jsd = calculate_jsd(chunk, reference_distribution);
            let analysis = JSDAnalysis { divergence: jsd };
            let [r, g, b] = analysis.to_color();
            Color32::from_rgb(r, g, b)
        })
        .collect()
}

// =============================================================================
// Multi-Scale Entropy (RCMSE) Visualization
// =============================================================================

/// Generate Refined Composite Multi-Scale Entropy (RCMSE) visualization.
///
/// Uses precomputed RCMSE values for fast rendering.
/// RCMSE reveals complexity across multiple time scales, distinguishing:
/// - Random/encrypted data (entropy decreases with scale)
/// - Structured/complex data (entropy constant across scales)
/// - Deterministic chaos (entropy increases then decreases)
///
/// Color legend:
/// - Purple/blue: Random/encrypted (steep negative slope)
/// - Cyan/teal: Compressed data (moderate negative slope)
/// - Green: Structured complexity (flat slope, 1/f-like)
/// - Yellow/orange: Chaotic/interesting patterns (positive early slope)
pub fn generate_rcmse_pixels(
    rcmse_map: &[f32],
    dimension: u64,
    file_size: u64,
    tex_size: usize,
    world_min: Vec2,
    scale_x: f32,
    scale_y: f32,
) -> Vec<Color32> {
    (0..tex_size * tex_size)
        .into_par_iter()
        .map(|idx| {
            let tex_y = idx / tex_size;
            let tex_x = idx % tex_size;

            let world_x = (world_min.x + tex_x as f32 * scale_x) as u64;
            let world_y = (world_min.y + tex_y as f32 * scale_y) as u64;

            if world_x >= dimension || world_y >= dimension {
                return Color32::from_rgb(13, 13, 13);
            }

            let d = xy2d(dimension, world_x, world_y);

            if d >= file_size {
                return Color32::from_rgb(13, 13, 13);
            }

            // Use precomputed RCMSE value (0.0-1.0 range)
            let rcmse_value = lookup_rcmse(rcmse_map, d as usize);

            // Map RCMSE value to color
            // 0.0-0.3: Random (purple/blue)
            // 0.3-0.6: Structured (green)
            // 0.6-1.0: Chaotic (yellow/orange)
            let t = rcmse_value.clamp(0.0, 1.0);

            let (r, g, b) = if t < 0.25 {
                // Random: Purple to blue
                let s = t / 0.25;
                (0.4 - s * 0.2, 0.1 + s * 0.2, 0.8 + s * 0.1)
            } else if t < 0.45 {
                // Transition: Blue to cyan/teal
                let s = (t - 0.25) / 0.2;
                (0.2 - s * 0.1, 0.3 + s * 0.3, 0.9 - s * 0.2)
            } else if t < 0.65 {
                // Structured: Cyan/teal to green
                let s = (t - 0.45) / 0.2;
                (0.1, 0.6 + s * 0.2, 0.7 - s * 0.4)
            } else if t < 0.85 {
                // Chaotic transition: Green to yellow
                let s = (t - 0.65) / 0.2;
                (0.1 + s * 0.8, 0.8, 0.3 - s * 0.2)
            } else {
                // Highly chaotic: Yellow to orange
                let s = (t - 0.85) / 0.15;
                (0.9 + s * 0.1, 0.8 - s * 0.2, 0.1)
            };

            Color32::from_rgb((r * 255.0) as u8, (g * 255.0) as u8, (b * 255.0) as u8)
        })
        .collect()
}

// =============================================================================
// Wavelet Entropy Visualization
// =============================================================================

/// Generate Wavelet Entropy visualization using Hilbert curve mapping.
///
/// Based on "Wavelet Decomposition of Software Entropy Reveals Symptoms of Malicious Code"
/// Uses precomputed wavelet suspiciousness values (SSECS) for fast rendering.
///
/// Key insight: Malware concentrates entropic energy at COARSE levels (large entropy shifts
/// from encrypted/compressed sections), while clean files concentrate energy at FINE levels.
///
/// Color legend:
/// - Deep blue/cyan: Low suspiciousness (normal, energy at fine levels)
/// - Green/teal: Medium-low suspiciousness
/// - Yellow/orange: Medium-high suspiciousness
/// - Red/magenta: High suspiciousness (malware-like, energy at coarse levels)
pub fn generate_wavelet_pixels(
    wavelet_map: &[f32],
    dimension: u64,
    file_size: u64,
    tex_size: usize,
    world_min: Vec2,
    scale_x: f32,
    scale_y: f32,
) -> Vec<Color32> {
    (0..tex_size * tex_size)
        .into_par_iter()
        .map(|idx| {
            let tex_y = idx / tex_size;
            let tex_x = idx % tex_size;

            let world_x = (world_min.x + tex_x as f32 * scale_x) as u64;
            let world_y = (world_min.y + tex_y as f32 * scale_y) as u64;

            if world_x >= dimension || world_y >= dimension {
                return Color32::from_rgb(13, 13, 13);
            }

            let d = xy2d(dimension, world_x, world_y);

            if d >= file_size {
                return Color32::from_rgb(13, 13, 13);
            }

            // Use precomputed wavelet suspiciousness value (0.0-1.0 range)
            let suspiciousness = lookup_wavelet(wavelet_map, d as usize);

            // Map suspiciousness to color using WaveletAnalysis color scheme
            let t = suspiciousness.clamp(0.0, 1.0);

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

            Color32::from_rgb((r * 255.0) as u8, (g * 255.0) as u8, (b * 255.0) as u8)
        })
        .collect()
}
