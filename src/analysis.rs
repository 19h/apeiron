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

/// Format data as a hex dump string.
///
/// Format: `XXXXXXXX  XX XX XX XX XX XX XX XX  XX XX XX XX XX XX XX XX  |................|`
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
}
