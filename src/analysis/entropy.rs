//! Shannon entropy calculation and byte distribution analysis.
//!
//! Provides entropy calculation, byte frequency distribution, and related utilities.
//! Optimized with:
//! - 4-way parallel histogram counting to avoid cache contention
//! - Cache-aligned buffers for optimal memory access
//! - Precomputed log2 lookup table for common count values
//! - SIMD-friendly loop structures

use rayon::prelude::*;

/// Window size for byte analysis (matches original implementation).
pub const ANALYSIS_WINDOW: usize = 64;

/// Cache-aligned histogram for optimal memory access.
/// 64-byte alignment ensures each histogram starts on a cache line boundary.
#[repr(C, align(64))]
struct AlignedHistogram {
    counts: [u32; 256],
}

impl AlignedHistogram {
    #[inline(always)]
    const fn new() -> Self {
        Self {
            counts: [0u32; 256],
        }
    }

    #[inline(always)]
    fn clear(&mut self) {
        // Use a simple loop - compiler optimizes this well
        for c in &mut self.counts {
            *c = 0;
        }
    }
}

/// Precomputed log2 values for counts 1..=LOG2_LUT_SIZE.
/// Avoids expensive log2 calls for common small counts.
const LOG2_LUT_SIZE: usize = 4096;

/// Generate log2 lookup table at compile time.
const fn generate_log2_lut() -> [f64; LOG2_LUT_SIZE + 1] {
    let mut lut = [0.0f64; LOG2_LUT_SIZE + 1];
    // lut[0] stays 0 (never used, but safe)
    let mut i = 1usize;
    while i <= LOG2_LUT_SIZE {
        // Manual log2 approximation for const context
        // log2(x) = ln(x) / ln(2)
        // We'll compute at runtime for accuracy, but have the array ready
        lut[i] = 0.0; // Placeholder - filled at first use
        i += 1;
    }
    lut
}

/// Runtime-initialized log2 lookup table.
/// Uses std::sync::OnceLock for thread-safe lazy initialization.
fn get_log2_lut() -> &'static [f64; LOG2_LUT_SIZE + 1] {
    use std::sync::OnceLock;
    static LOG2_LUT: OnceLock<[f64; LOG2_LUT_SIZE + 1]> = OnceLock::new();
    LOG2_LUT.get_or_init(|| {
        let mut lut = [0.0f64; LOG2_LUT_SIZE + 1];
        for i in 1..=LOG2_LUT_SIZE {
            lut[i] = (i as f64).log2();
        }
        lut
    })
}

/// Fast log2 using lookup table for small values, computed for large values.
#[inline(always)]
fn fast_log2(x: u32) -> f64 {
    if x == 0 {
        return 0.0; // Avoid special case in caller
    }
    if x as usize <= LOG2_LUT_SIZE {
        get_log2_lut()[x as usize]
    } else {
        (x as f64).log2()
    }
}

/// Count bytes using 4-way parallel histograms.
/// This avoids cache line contention when processing sequential bytes
/// that might hash to the same histogram buckets.
#[inline]
fn count_bytes_4way(data: &[u8], out: &mut [u32; 256]) {
    // Four separate histograms to avoid cache contention
    let mut h0 = AlignedHistogram::new();
    let mut h1 = AlignedHistogram::new();
    let mut h2 = AlignedHistogram::new();
    let mut h3 = AlignedHistogram::new();

    // Process 4 bytes at a time
    let chunks = data.chunks_exact(4);
    let remainder = chunks.remainder();

    for chunk in chunks {
        // Safety: chunks_exact guarantees 4 bytes
        unsafe {
            let b0 = *chunk.get_unchecked(0) as usize;
            let b1 = *chunk.get_unchecked(1) as usize;
            let b2 = *chunk.get_unchecked(2) as usize;
            let b3 = *chunk.get_unchecked(3) as usize;
            *h0.counts.get_unchecked_mut(b0) += 1;
            *h1.counts.get_unchecked_mut(b1) += 1;
            *h2.counts.get_unchecked_mut(b2) += 1;
            *h3.counts.get_unchecked_mut(b3) += 1;
        }
    }

    // Handle remainder
    for (i, &byte) in remainder.iter().enumerate() {
        match i {
            0 => h0.counts[byte as usize] += 1,
            1 => h1.counts[byte as usize] += 1,
            2 => h2.counts[byte as usize] += 1,
            _ => h3.counts[byte as usize] += 1,
        }
    }

    // Merge histograms - process 4 at a time for better vectorization
    let chunks = out.chunks_exact_mut(4);
    let mut i = 0;
    for chunk in chunks {
        chunk[0] = h0.counts[i] + h1.counts[i] + h2.counts[i] + h3.counts[i];
        chunk[1] = h0.counts[i + 1] + h1.counts[i + 1] + h2.counts[i + 1] + h3.counts[i + 1];
        chunk[2] = h0.counts[i + 2] + h1.counts[i + 2] + h2.counts[i + 2] + h3.counts[i + 2];
        chunk[3] = h0.counts[i + 3] + h1.counts[i + 3] + h2.counts[i + 3] + h3.counts[i + 3];
        i += 4;
    }
}

/// Count bytes using simple sequential method (better for small data).
#[inline]
fn count_bytes_simple(data: &[u8], out: &mut [u32; 256]) {
    for c in out.iter_mut() {
        *c = 0;
    }
    for &byte in data {
        out[byte as usize] += 1;
    }
}

/// Threshold for using 4-way counting vs simple counting.
/// Below this, the setup overhead of 4-way counting isn't worth it.
const FOURWAY_THRESHOLD: usize = 256;

/// Calculate Shannon entropy for a byte slice.
///
/// Shannon entropy measures the average information content per byte.
/// Values range from 0 (completely uniform) to 8 (maximum randomness).
pub fn calculate_entropy(data: &[u8]) -> f64 {
    if data.is_empty() {
        return 0.0;
    }

    let mut counts = [0u32; 256];

    if data.len() >= FOURWAY_THRESHOLD {
        count_bytes_4way(data, &mut counts);
    } else {
        count_bytes_simple(data, &mut counts);
    }

    let total = data.len() as f64;
    let inv_total = 1.0 / total;

    // Compute entropy using optimized log2
    // H = -sum(p * log2(p)) = -sum((c/n) * log2(c/n))
    //   = -sum((c/n) * (log2(c) - log2(n)))
    //   = log2(n) - (1/n) * sum(c * log2(c))
    let log2_total = total.log2();
    let mut sum_c_log_c = 0.0f64;

    // Process 4 counts at a time for better pipelining
    let chunks = counts.chunks_exact(4);
    for chunk in chunks {
        // Branchless: multiply by 0 if count is 0
        let c0 = chunk[0] as f64;
        let c1 = chunk[1] as f64;
        let c2 = chunk[2] as f64;
        let c3 = chunk[3] as f64;

        if chunk[0] > 0 {
            sum_c_log_c += c0 * fast_log2(chunk[0]);
        }
        if chunk[1] > 0 {
            sum_c_log_c += c1 * fast_log2(chunk[1]);
        }
        if chunk[2] > 0 {
            sum_c_log_c += c2 * fast_log2(chunk[2]);
        }
        if chunk[3] > 0 {
            sum_c_log_c += c3 * fast_log2(chunk[3]);
        }
    }

    log2_total - inv_total * sum_c_log_c
}

/// Optimized entropy calculation for a chunk (typically 256 bytes).
/// Uses precomputed reciprocal and avoids branching where possible.
#[inline(always)]
pub fn chunk_entropy_fast(chunk: &[u8]) -> f64 {
    debug_assert!(!chunk.is_empty());

    let mut counts = [0u32; 256];

    // For typical chunk sizes (64-256 bytes), simple counting is faster
    if chunk.len() >= FOURWAY_THRESHOLD {
        count_bytes_4way(chunk, &mut counts);
    } else {
        count_bytes_simple(chunk, &mut counts);
    }

    let total = chunk.len() as f64;
    let inv_total = 1.0 / total;
    let log2_total = total.log2();

    // H = log2(n) - (1/n) * sum(c * log2(c))
    let mut sum_c_log_c = 0.0f64;

    for &count in &counts {
        if count > 0 {
            let c = count as f64;
            sum_c_log_c += c * fast_log2(count);
        }
    }

    log2_total - inv_total * sum_c_log_c
}

/// Calculate byte frequency distribution (probability distribution) for a byte slice.
///
/// Returns an array of 256 probabilities, one for each possible byte value.
pub fn byte_distribution(data: &[u8]) -> [f64; 256] {
    if data.is_empty() {
        // Return uniform distribution for empty data
        return [1.0 / 256.0; 256];
    }

    let mut counts = [0u32; 256];

    if data.len() >= FOURWAY_THRESHOLD {
        count_bytes_4way(data, &mut counts);
    } else {
        count_bytes_simple(data, &mut counts);
    }

    let inv_total = 1.0 / data.len() as f64;
    let mut dist = [0.0f64; 256];

    // Convert counts to probabilities - process 4 at a time
    for i in (0..256).step_by(4) {
        dist[i] = counts[i] as f64 * inv_total;
        dist[i + 1] = counts[i + 1] as f64 * inv_total;
        dist[i + 2] = counts[i + 2] as f64 * inv_total;
        dist[i + 3] = counts[i + 3] as f64 * inv_total;
    }

    dist
}

/// Calculate byte frequency distribution using sampling for large files.
///
/// For files smaller than 10MB, this uses the full distribution.
/// For larger files, it samples strategically to avoid O(n) iteration:
/// - First 1MB is fully counted (headers often have distinct patterns)
/// - Remaining file is sampled at regular intervals
///
/// This provides an accurate distribution estimate while being O(1) for large files.
pub fn byte_distribution_sampled(data: &[u8]) -> [f64; 256] {
    const SMALL_FILE_THRESHOLD: usize = 10 * 1024 * 1024; // 10MB
    const HEADER_SIZE: usize = 1024 * 1024; // 1MB of header to fully count
    const TARGET_SAMPLES: usize = 10_000_000; // ~10 million samples from body

    if data.is_empty() {
        return [1.0 / 256.0; 256];
    }

    // For small files, use full distribution
    if data.len() <= SMALL_FILE_THRESHOLD {
        return byte_distribution(data);
    }

    let mut counts = [0u32; 256];
    let mut sample_count = 0u64;

    // Count all bytes in the header (first 1MB) using optimized counting
    let header_end = HEADER_SIZE.min(data.len());
    if header_end >= FOURWAY_THRESHOLD {
        count_bytes_4way(&data[..header_end], &mut counts);
    } else {
        count_bytes_simple(&data[..header_end], &mut counts);
    }
    sample_count += header_end as u64;

    // Sample the rest of the file
    if data.len() > HEADER_SIZE {
        let body = &data[HEADER_SIZE..];
        let body_len = body.len();

        // Calculate step size to get ~TARGET_SAMPLES from the body
        let step = (body_len / TARGET_SAMPLES).max(1);

        // Use strided access for sampling
        let mut offset = 0;
        while offset < body_len {
            // Safety: offset is always < body_len
            unsafe {
                let byte = *body.get_unchecked(offset);
                *counts.get_unchecked_mut(byte as usize) += 1;
            }
            sample_count += 1;
            offset += step;
        }
    }

    // Convert counts to probabilities
    let inv_total = 1.0 / sample_count as f64;
    let mut dist = [0.0f64; 256];
    for i in 0..256 {
        dist[i] = counts[i] as f64 * inv_total;
    }
    dist
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

/// Extract the first printable ASCII string (length > 4) from data.
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

/// Compute entropy into a preallocated counts buffer (avoids allocation).
/// Returns entropy value.
#[inline]
pub fn calculate_entropy_with_buffer(data: &[u8], counts: &mut [u32; 256]) -> f64 {
    if data.is_empty() {
        return 0.0;
    }

    // Clear buffer
    for c in counts.iter_mut() {
        *c = 0;
    }

    if data.len() >= FOURWAY_THRESHOLD {
        count_bytes_4way(data, counts);
    } else {
        count_bytes_simple(data, counts);
    }

    let total = data.len() as f64;
    let inv_total = 1.0 / total;
    let log2_total = total.log2();
    let mut sum_c_log_c = 0.0f64;

    for &count in counts.iter() {
        if count > 0 {
            let c = count as f64;
            sum_c_log_c += c * fast_log2(count);
        }
    }

    log2_total - inv_total * sum_c_log_c
}

/// Calculate entropy for multiple windows in a batch.
/// This is more efficient than calling calculate_entropy repeatedly because:
/// - Allocates a single reusable histogram buffer
/// - Better cache locality when processing sequential windows
/// - Amortizes function call overhead
///
/// # Arguments
/// * `data` - The full data buffer
/// * `offsets` - Starting positions for each window
/// * `window_size` - Size of each window
///
/// # Returns
/// Vector of entropy values, one for each offset
pub fn calculate_entropy_batch(data: &[u8], offsets: &[usize], window_size: usize) -> Vec<f64> {
    // For small batches, just use normal entropy calculation
    if offsets.len() < 64 {
        return offsets
            .iter()
            .map(|&offset| {
                let end = (offset + window_size).min(data.len());
                if offset < data.len() && end > offset {
                    calculate_entropy(&data[offset..end])
                } else {
                    0.0
                }
            })
            .collect();
    }

    // For larger batches, use parallel processing with thread-local buffers
    offsets
        .par_iter()
        .map(|&offset| {
            let end = (offset + window_size).min(data.len());
            if offset < data.len() && end > offset {
                // Thread-local would be ideal but for simplicity use inline buffer
                let mut counts = [0u32; 256];
                calculate_entropy_with_buffer(&data[offset..end], &mut counts)
            } else {
                0.0
            }
        })
        .collect()
}

/// Calculate entropy for multiple windows, returning results in a preallocated buffer.
/// Even more efficient than calculate_entropy_batch when called repeatedly.
#[inline]
pub fn calculate_entropy_batch_into(
    data: &[u8],
    offsets: &[usize],
    window_size: usize,
    results: &mut [f64],
) {
    debug_assert!(results.len() >= offsets.len());

    // Sequential processing with reused histogram buffer for small batches
    if offsets.len() < 64 {
        let mut counts = [0u32; 256];
        for (i, &offset) in offsets.iter().enumerate() {
            let end = (offset + window_size).min(data.len());
            results[i] = if offset < data.len() && end > offset {
                calculate_entropy_with_buffer(&data[offset..end], &mut counts)
            } else {
                0.0
            };
        }
        return;
    }

    // Parallel processing for large batches
    let entropies = calculate_entropy_batch(data, offsets, window_size);
    results[..entropies.len()].copy_from_slice(&entropies);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_entropy_uniform() {
        // All zeros should have zero entropy
        let data = vec![0u8; 256];
        assert!(calculate_entropy(&data) < 0.01);
    }

    #[test]
    fn test_entropy_random() {
        // All different bytes should have maximum entropy (8 bits)
        let data: Vec<u8> = (0..=255).collect();
        let entropy = calculate_entropy(&data);
        assert!((entropy - 8.0).abs() < 0.01);
    }

    #[test]
    fn test_byte_distribution() {
        let data = vec![0u8, 1, 2, 3];
        let dist = byte_distribution(&data);
        assert!((dist[0] - 0.25).abs() < 0.01);
        assert!((dist[1] - 0.25).abs() < 0.01);
        assert!(dist[255] < 0.01);
    }

    #[test]
    fn test_extract_ascii() {
        let data = b"Hello World\x00\xff\xfe";
        let ascii = extract_ascii(data);
        assert_eq!(ascii, Some("Hello World".to_string()));
    }

    #[test]
    fn test_fourway_counting_correctness() {
        // Verify 4-way counting produces same results as simple
        let data: Vec<u8> = (0..1000).map(|i| (i * 17 + 31) as u8).collect();

        let mut counts_simple = [0u32; 256];
        count_bytes_simple(&data, &mut counts_simple);

        let mut counts_4way = [0u32; 256];
        count_bytes_4way(&data, &mut counts_4way);

        for i in 0..256 {
            assert_eq!(counts_simple[i], counts_4way[i], "Mismatch at index {i}");
        }
    }

    #[test]
    fn test_large_data_entropy() {
        // Test with larger data to exercise 4-way path
        let data: Vec<u8> = (0..10000).map(|i| (i % 256) as u8).collect();
        let entropy = calculate_entropy(&data);
        // Should be close to max entropy since we have uniform distribution
        assert!(entropy > 7.9);
    }
}
