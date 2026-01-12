//! Shannon entropy calculation and byte distribution analysis.
//!
//! Provides entropy calculation, byte frequency distribution, and related utilities.

use rayon::prelude::*;

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

/// Optimized entropy calculation for a chunk (typically 256 bytes).
/// Uses precomputed reciprocal and avoids branching where possible.
#[inline(always)]
pub fn chunk_entropy_fast(chunk: &[u8]) -> f64 {
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

    let mut counts = [0u64; 256];
    let mut sample_count = 0u64;

    // Count all bytes in the header (first 1MB)
    let header_end = HEADER_SIZE.min(data.len());
    for &byte in &data[..header_end] {
        counts[byte as usize] += 1;
        sample_count += 1;
    }

    // Sample the rest of the file
    if data.len() > HEADER_SIZE {
        let body = &data[HEADER_SIZE..];
        let body_len = body.len();

        // Calculate step size to get ~TARGET_SAMPLES from the body
        let step = (body_len / TARGET_SAMPLES).max(1);

        let mut offset = 0;
        while offset < body_len {
            counts[body[offset] as usize] += 1;
            sample_count += 1;
            offset += step;
        }
    }

    // Convert counts to probabilities
    let total = sample_count as f64;
    let mut dist = [0.0f64; 256];
    for i in 0..256 {
        dist[i] = counts[i] as f64 / total;
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
}
