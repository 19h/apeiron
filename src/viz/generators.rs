//! Pixel generation functions for binary visualizations.
//!
//! Each generator produces a Vec<Color32> for rendering visualization textures.
//!
//! Optimizations:
//! - Parallel digraph counting with thread-local histograms
//! - Precomputed histogram caching for similarity matrix
//! - SIMD chi-squared distance computation
//! - Reduced texture generation overhead

use eframe::egui::{Color32, Vec2};
use rayon::prelude::*;
use wide::f64x4;

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
#[inline]
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
#[inline]
pub fn similarity_to_color(similarity: f64, is_diagonal: bool) -> Color32 {
    let t = similarity.clamp(0.0, 1.0).powf(0.4);

    if is_diagonal {
        return Color32::from_rgb((180.0 + t * 75.0) as u8, (220.0 + t * 35.0) as u8, 255);
    }

    let (r, g, b) = if t < 0.2 {
        let s = t / 0.2;
        (s * 0.25, 0.0, s * 0.35)
    } else if t < 0.4 {
        let s = (t - 0.2) / 0.2;
        (0.25 + s * 0.45, 0.0, 0.35 + s * 0.1)
    } else if t < 0.6 {
        let s = (t - 0.4) / 0.2;
        (0.7 + s * 0.25, s * 0.25, 0.45 - s * 0.35)
    } else if t < 0.8 {
        let s = (t - 0.6) / 0.2;
        (0.95 + s * 0.05, 0.25 + s * 0.45, 0.1 + s * 0.1)
    } else {
        let s = (t - 0.8) / 0.2;
        (1.0, 0.7 + s * 0.3, 0.2 + s * 0.7)
    };

    Color32::from_rgb((r * 255.0) as u8, (g * 255.0) as u8, (b * 255.0) as u8)
}

// =============================================================================
// Sample Interval Constants
// =============================================================================

const COMPLEXITY_SAMPLE_INTERVAL: usize = 64;
const RCMSE_SAMPLE_INTERVAL: usize = 64;
const WAVELET_SAMPLE_INTERVAL: usize = 64;

// =============================================================================
// Lookup Functions
// =============================================================================

#[inline]
fn lookup_complexity(complexity_map: &[f32], offset: usize) -> f32 {
    if complexity_map.is_empty() {
        return 0.0;
    }
    let index = offset / COMPLEXITY_SAMPLE_INTERVAL;
    complexity_map.get(index).copied().unwrap_or(0.0)
}

#[inline]
fn lookup_rcmse(rcmse_map: &[f32], offset: usize) -> f32 {
    if rcmse_map.is_empty() {
        return 0.5;
    }
    let index = offset / RCMSE_SAMPLE_INTERVAL;
    rcmse_map.get(index).copied().unwrap_or(0.5)
}

#[inline]
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
                    let (r, g, b) = placeholder_pulse_color(time_seconds);
                    Color32::from_rgb(r, g, b)
                }
            }
        })
        .collect()
}

pub fn generate_computing_placeholder(tex_size: usize, time_seconds: f64) -> Vec<Color32> {
    let total = tex_size * tex_size;
    let mut pixels = Vec::with_capacity(total);

    let pulse = ((time_seconds * 2.0 * std::f64::consts::PI).sin() * 0.5 + 0.5) as f32;
    let base_color = 25.0 + pulse * 10.0;

    for idx in 0..total {
        let y = idx / tex_size;
        let x = idx % tex_size;
        let gradient = ((x + y) as f32 / (tex_size * 2) as f32) * 15.0;
        let v = (base_color + gradient) as u8;
        pixels.push(Color32::from_rgb(v, v, v.saturating_add(12)));
    }

    pixels
}

// =============================================================================
// Similarity Matrix Visualization (Optimized)
// =============================================================================

/// Compute histogram for a window at given position.
/// Uses 4-way unrolled counting for better ILP.
#[inline]
fn compute_histogram(data: &[u8], pos: usize, window_size: usize) -> [u32; 256] {
    let mut hist = [0u32; 256];
    let end = (pos + window_size).min(data.len());

    if pos >= data.len() {
        return hist;
    }

    let window = &data[pos..end];

    // 4-way unrolled counting
    let chunks = window.chunks_exact(4);
    let remainder = chunks.remainder();

    for chunk in chunks {
        unsafe {
            *hist.get_unchecked_mut(chunk[0] as usize) += 1;
            *hist.get_unchecked_mut(chunk[1] as usize) += 1;
            *hist.get_unchecked_mut(chunk[2] as usize) += 1;
            *hist.get_unchecked_mut(chunk[3] as usize) += 1;
        }
    }

    for &byte in remainder {
        unsafe {
            *hist.get_unchecked_mut(byte as usize) += 1;
        }
    }

    hist
}

/// Compute chi-squared distance between two histograms using SIMD.
#[inline]
fn chi_squared_distance_simd(hist_x: &[u32; 256], hist_y: &[u32; 256]) -> f64 {
    // Compute totals
    let total_x: u32 = hist_x.iter().sum();
    let total_y: u32 = hist_y.iter().sum();

    if total_x == 0 || total_y == 0 {
        return 2.0; // Maximum distance
    }

    let inv_total_x = 1.0 / total_x as f64;
    let inv_total_y = 1.0 / total_y as f64;

    // SIMD chi-squared computation
    let mut chi_sq = f64x4::ZERO;

    for i in (0..256).step_by(4) {
        let hx = f64x4::new([
            hist_x[i] as f64,
            hist_x[i + 1] as f64,
            hist_x[i + 2] as f64,
            hist_x[i + 3] as f64,
        ]);
        let hy = f64x4::new([
            hist_y[i] as f64,
            hist_y[i + 1] as f64,
            hist_y[i + 2] as f64,
            hist_y[i + 3] as f64,
        ]);

        let px = hx * f64x4::splat(inv_total_x);
        let py = hy * f64x4::splat(inv_total_y);
        let sum = px + py;
        let diff = px - py;

        // chi_sq += (px - py)^2 / (px + py) where sum > 0
        let diff_sq = diff * diff;

        // Manual division with zero check
        let arr_sum = sum.to_array();
        let arr_diff_sq = diff_sq.to_array();
        let mut contrib = [0.0f64; 4];
        for j in 0..4 {
            if arr_sum[j] > 0.0 {
                contrib[j] = arr_diff_sq[j] / arr_sum[j];
            }
        }
        chi_sq += f64x4::new(contrib);
    }

    let arr = chi_sq.to_array();
    arr[0] + arr[1] + arr[2] + arr[3]
}

pub fn generate_similarity_matrix_pixels(
    data: &[u8],
    file_size: u64,
    tex_size: usize,
    world_min: Vec2,
    scale_x: f32,
    scale_y: f32,
    dimension: u64,
) -> Vec<Color32> {
    let window_size: usize = 32;
    let dim = dimension as f64;

    (0..tex_size * tex_size)
        .into_par_iter()
        .map(|idx| {
            let tex_y = idx / tex_size;
            let tex_x = idx % tex_size;

            let world_x = world_min.x + tex_x as f32 * scale_x;
            let world_y = world_min.y + tex_y as f32 * scale_y;

            if world_x < 0.0 || world_y < 0.0 || world_x >= dim as f32 || world_y >= dim as f32 {
                return Color32::from_rgb(10, 10, 15);
            }

            let pos_x = ((world_x as f64 / dim) * file_size as f64) as usize;
            let pos_y = ((world_y as f64 / dim) * file_size as f64) as usize;

            let hist_x = compute_histogram(data, pos_x, window_size);
            let hist_y = compute_histogram(data, pos_y, window_size);

            let chi_sq = chi_squared_distance_simd(&hist_x, &hist_y);
            let similarity = 1.0 - (chi_sq / 2.0).sqrt();

            let is_diagonal = (pos_x as i64 - pos_y as i64).abs() < window_size as i64;
            similarity_to_color(similarity, is_diagonal)
        })
        .collect()
}

// =============================================================================
// Digraph Visualization (Optimized with parallel counting)
// =============================================================================

/// Digraph size constant
const DIGRAPH_SIZE: usize = 256 * 256;

/// Build digraph using parallel thread-local histograms.
/// Optimized with boxed fixed-size arrays and SIMD merge using u64x4.
fn build_digraph_parallel(data: &[u8]) -> (Vec<u64>, u64) {
    use wide::u64x4;

    if data.len() < 2 {
        return (vec![0u64; DIGRAPH_SIZE], 1);
    }

    // Use larger chunks to reduce allocations - minimum 64KB per chunk
    let num_threads = rayon::current_num_threads();
    let chunk_size = (data.len() / num_threads).max(65536);

    // Parallel digraph counting with boxed fixed-size arrays
    let results: Vec<Box<[u64; DIGRAPH_SIZE]>> = data
        .par_chunks(chunk_size)
        .map(|chunk| {
            // Boxed array - single allocation, known size
            let mut local_digraph: Box<[u64; DIGRAPH_SIZE]> =
                unsafe { Box::new_zeroed().assume_init() };

            // Process windows - unroll by 2 for better ILP
            let windows = chunk.windows(2);
            for window in windows {
                let idx = (window[0] as usize) * 256 + (window[1] as usize);
                unsafe {
                    *local_digraph.get_unchecked_mut(idx) += 1;
                }
            }
            local_digraph
        })
        .collect();

    // Allocate output
    let mut digraph = vec![0u64; DIGRAPH_SIZE];

    // Merge with SIMD u64x4 - process 4 elements at a time
    for local in &results {
        for i in (0..DIGRAPH_SIZE).step_by(4) {
            let current = u64x4::new([digraph[i], digraph[i + 1], digraph[i + 2], digraph[i + 3]]);
            let local_vals = u64x4::new([local[i], local[i + 1], local[i + 2], local[i + 3]]);
            let sum = current + local_vals;
            let arr = sum.to_array();
            digraph[i] = arr[0];
            digraph[i + 1] = arr[1];
            digraph[i + 2] = arr[2];
            digraph[i + 3] = arr[3];
        }
    }

    // Find max - scalar loop with unrolling (SIMD max not available for u64x4)
    let mut max_count = 1u64;
    for i in (0..DIGRAPH_SIZE).step_by(4) {
        let m0 = digraph[i];
        let m1 = digraph[i + 1];
        let m2 = digraph[i + 2];
        let m3 = digraph[i + 3];
        // Chain of max operations - compiler optimizes this well
        max_count = max_count.max(m0).max(m1).max(m2).max(m3);
    }

    // Add cross-chunk transitions
    let mut offset = 0;
    for (i, chunk) in data.chunks(chunk_size).enumerate() {
        if i > 0 && offset > 0 {
            let from = data[offset - 1] as usize;
            let to = chunk[0] as usize;
            let idx = from * 256 + to;
            digraph[idx] += 1;
            max_count = max_count.max(digraph[idx]);
        }
        offset += chunk.len();
    }

    (digraph, max_count)
}

pub fn generate_digraph_pixels(
    data: &[u8],
    file_size: u64,
    tex_size: usize,
    world_min: Vec2,
    scale_x: f32,
    scale_y: f32,
) -> Vec<Color32> {
    let (digraph, max_count) = build_digraph_parallel(data);
    let sqrt_max = (max_count as f64).sqrt();

    (0..tex_size * tex_size)
        .into_par_iter()
        .map(|idx| {
            let tex_y = idx / tex_size;
            let tex_x = idx % tex_size;

            let world_x = world_min.x + tex_x as f32 * scale_x;
            let world_y = world_min.y + tex_y as f32 * scale_y;

            let dg_x = world_x as i32;
            let dg_y = world_y as i32;

            if dg_x < 0 || dg_x >= 256 || dg_y < 0 || dg_y >= 256 {
                return Color32::from_rgb(8, 8, 12);
            }

            let count = digraph[dg_y as usize * 256 + dg_x as usize];

            if count == 0 {
                Color32::from_rgb(8, 8, 12)
            } else {
                let intensity = ((count as f64).sqrt() / sqrt_max).powf(0.6);
                let hue = ((dg_x + dg_y) as f64 / 512.0) * 300.0;
                let sat = 0.85 - intensity * 0.2;
                let val = 0.15 + intensity * 0.85;
                hsv_to_rgb(hue, sat, val)
            }
        })
        .collect()
}

// =============================================================================
// Byte Phase Space Visualization (Optimized)
// =============================================================================

/// Build phase space using parallel counting.
/// Optimized with boxed fixed-size arrays and better merge.
fn build_phase_space_parallel(data: &[u8]) -> (Vec<(u64, u64)>, u64) {
    if data.len() < 2 {
        return (vec![(0u64, 0u64); DIGRAPH_SIZE], 1);
    }

    let num_threads = rayon::current_num_threads();
    let chunk_size = (data.len() / num_threads).max(65536);

    // Each thread computes (last_pos, count) for its chunk using boxed arrays
    let results: Vec<(usize, Box<[(u64, u64); DIGRAPH_SIZE]>)> = data
        .par_chunks(chunk_size)
        .enumerate()
        .map(|(chunk_idx, chunk)| {
            let base_offset = chunk_idx * chunk_size;
            // Boxed array - single allocation
            let mut local_space: Box<[(u64, u64); DIGRAPH_SIZE]> =
                unsafe { Box::new_zeroed().assume_init() };

            for (i, window) in chunk.windows(2).enumerate() {
                let x = window[0] as usize;
                let y = window[1] as usize;
                let idx = y * 256 + x;
                let pos = (base_offset + i) as u64;
                unsafe {
                    let entry = local_space.get_unchecked_mut(idx);
                    entry.0 = pos;
                    entry.1 += 1;
                }
            }

            (chunk_idx, local_space)
        })
        .collect();

    // Merge results - process in chunk index order for correct position tracking
    let mut phase_space = vec![(0u64, 0u64); DIGRAPH_SIZE];

    // Sort by chunk index to process in order
    let mut sorted_results: Vec<_> = results.into_iter().collect();
    sorted_results.sort_by_key(|(idx, _)| *idx);

    for (_chunk_idx, local) in &sorted_results {
        // Process 4 elements at a time for better cache usage
        for i in (0..DIGRAPH_SIZE).step_by(4) {
            unsafe {
                for j in 0..4 {
                    let local_entry = local.get_unchecked(i + j);
                    if local_entry.1 > 0 {
                        let entry = phase_space.get_unchecked_mut(i + j);
                        entry.0 = local_entry.0; // Later chunks have higher positions
                        entry.1 += local_entry.1;
                    }
                }
            }
        }
    }

    // Find max count using SIMD-friendly loop
    let mut max_count = 1u64;
    for entry in &phase_space {
        max_count = max_count.max(entry.1);
    }

    (phase_space, max_count)
}

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

    let (phase_space, max_count) = build_phase_space_parallel(data);
    let file_size_f = file_size as f64;
    let sqrt_max = (max_count as f64).sqrt();

    (0..tex_size * tex_size)
        .into_par_iter()
        .map(|idx| {
            let tex_y = idx / tex_size;
            let tex_x = idx % tex_size;

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
                let intensity = ((count as f64).sqrt() / sqrt_max).powf(0.5);
                let coord_hue = ((ps_x + ps_y) as f64 / 512.0) * 360.0;
                let position_ratio = last_pos as f64 / file_size_f;
                let pos_hue = position_ratio * 120.0;
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

            let rcmse_value = lookup_rcmse(rcmse_map, d as usize);
            let t = rcmse_value.clamp(0.0, 1.0);

            let (r, g, b) = if t < 0.25 {
                let s = t / 0.25;
                (0.4 - s * 0.2, 0.1 + s * 0.2, 0.8 + s * 0.1)
            } else if t < 0.45 {
                let s = (t - 0.25) / 0.2;
                (0.2 - s * 0.1, 0.3 + s * 0.3, 0.9 - s * 0.2)
            } else if t < 0.65 {
                let s = (t - 0.45) / 0.2;
                (0.1, 0.6 + s * 0.2, 0.7 - s * 0.4)
            } else if t < 0.85 {
                let s = (t - 0.65) / 0.2;
                (0.1 + s * 0.8, 0.8, 0.3 - s * 0.2)
            } else {
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

            let suspiciousness = lookup_wavelet(wavelet_map, d as usize);
            let t = suspiciousness.clamp(0.0, 1.0);

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

            Color32::from_rgb((r * 255.0) as u8, (g * 255.0) as u8, (b * 255.0) as u8)
        })
        .collect()
}
