//! Byte-level analysis and file type detection.
//!
//! Provides forensic byte analysis and magic byte detection.
//!
//! Optimizations:
//! - 256-byte lookup table for byte classification
//! - SIMD u32x8 for parallel flag accumulation
//! - SIMD i16x8 for parallel variation calculation
//! - 8-way unrolled main loop for maximum ILP

use wide::{i16x8, u32x4};

/// Byte classification flags (packed into u8 for cache efficiency).
/// Bit 0: Is printable ASCII (32-126)
/// Bit 1: Is high-bit (> 127)
/// Bit 2: Is null (0)
const FLAG_TEXT: u8 = 0b001;
const FLAG_HIGH: u8 = 0b010;
const FLAG_NULL: u8 = 0b100;

/// Precomputed byte classification lookup table.
/// Each byte maps to its classification flags.
const fn generate_byte_class_lut() -> [u8; 256] {
    let mut lut = [0u8; 256];
    let mut i = 0usize;
    while i < 256 {
        let mut flags = 0u8;
        // Is printable ASCII (32-126)?
        if i >= 32 && i <= 126 {
            flags |= FLAG_TEXT;
        }
        // Is high-bit (> 127)?
        if i > 127 {
            flags |= FLAG_HIGH;
        }
        // Is null (0)?
        if i == 0 {
            flags |= FLAG_NULL;
        }
        lut[i] = flags;
        i += 1;
    }
    lut
}

/// Static byte classification lookup table (computed at compile time).
static BYTE_CLASS_LUT: [u8; 256] = generate_byte_class_lut();

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
    /// Optimized with SIMD accumulation and 8-way unrolling.
    pub fn analyze(data: &[u8]) -> Self {
        if data.is_empty() {
            return Self {
                text_ratio: 0.0,
                high_ratio: 0.0,
                null_ratio: 0.0,
                variation: 0.0,
            };
        }

        let mut text_count = 0u32;
        let mut high_count = 0u32;
        let mut null_count = 0u32;
        let mut variation = 0u64; // Use u64 to avoid overflow with large files

        // Process bytes using SIMD-accelerated 8-way unrolling
        let chunks = data.chunks_exact(8);
        let remainder = chunks.remainder();

        let mut prev_byte = data[0] as i16;

        for chunk in chunks {
            // Lookup classifications for all 8 bytes
            let c0 = BYTE_CLASS_LUT[chunk[0] as usize];
            let c1 = BYTE_CLASS_LUT[chunk[1] as usize];
            let c2 = BYTE_CLASS_LUT[chunk[2] as usize];
            let c3 = BYTE_CLASS_LUT[chunk[3] as usize];
            let c4 = BYTE_CLASS_LUT[chunk[4] as usize];
            let c5 = BYTE_CLASS_LUT[chunk[5] as usize];
            let c6 = BYTE_CLASS_LUT[chunk[6] as usize];
            let c7 = BYTE_CLASS_LUT[chunk[7] as usize];

            // Accumulate flags using SIMD u32x4 operations
            // First 4 bytes
            let flags0 = u32x4::new([c0 as u32, c1 as u32, c2 as u32, c3 as u32]);
            // Second 4 bytes
            let flags1 = u32x4::new([c4 as u32, c5 as u32, c6 as u32, c7 as u32]);

            // Extract text flags (bit 0)
            let text_mask = u32x4::splat(FLAG_TEXT as u32);
            let text0 = flags0 & text_mask;
            let text1 = flags1 & text_mask;
            let text_sum0 = text0.to_array();
            let text_sum1 = text1.to_array();
            text_count += text_sum0[0]
                + text_sum0[1]
                + text_sum0[2]
                + text_sum0[3]
                + text_sum1[0]
                + text_sum1[1]
                + text_sum1[2]
                + text_sum1[3];

            // Extract high flags (bit 1) - extract array and shift scalarly
            let f0_arr = flags0.to_array();
            let f1_arr = flags1.to_array();
            high_count += ((f0_arr[0] & (FLAG_HIGH as u32)) >> 1)
                + ((f0_arr[1] & (FLAG_HIGH as u32)) >> 1)
                + ((f0_arr[2] & (FLAG_HIGH as u32)) >> 1)
                + ((f0_arr[3] & (FLAG_HIGH as u32)) >> 1)
                + ((f1_arr[0] & (FLAG_HIGH as u32)) >> 1)
                + ((f1_arr[1] & (FLAG_HIGH as u32)) >> 1)
                + ((f1_arr[2] & (FLAG_HIGH as u32)) >> 1)
                + ((f1_arr[3] & (FLAG_HIGH as u32)) >> 1);

            // Extract null flags (bit 2)
            null_count += ((f0_arr[0] & (FLAG_NULL as u32)) >> 2)
                + ((f0_arr[1] & (FLAG_NULL as u32)) >> 2)
                + ((f0_arr[2] & (FLAG_NULL as u32)) >> 2)
                + ((f0_arr[3] & (FLAG_NULL as u32)) >> 2)
                + ((f1_arr[0] & (FLAG_NULL as u32)) >> 2)
                + ((f1_arr[1] & (FLAG_NULL as u32)) >> 2)
                + ((f1_arr[2] & (FLAG_NULL as u32)) >> 2)
                + ((f1_arr[3] & (FLAG_NULL as u32)) >> 2);

            // SIMD variation calculation using i16x8
            let bytes = i16x8::new([
                chunk[0] as i16,
                chunk[1] as i16,
                chunk[2] as i16,
                chunk[3] as i16,
                chunk[4] as i16,
                chunk[5] as i16,
                chunk[6] as i16,
                chunk[7] as i16,
            ]);
            let prev = i16x8::new([
                prev_byte,
                chunk[0] as i16,
                chunk[1] as i16,
                chunk[2] as i16,
                chunk[3] as i16,
                chunk[4] as i16,
                chunk[5] as i16,
                chunk[6] as i16,
            ]);

            // Compute absolute differences using saturating subtraction trick
            // |a - b| = max(a - b, b - a)
            let diff1 = bytes - prev;
            let diff2 = prev - bytes;
            let arr1 = diff1.to_array();
            let arr2 = diff2.to_array();

            // Compute absolute values and sum
            for i in 0..8 {
                variation += arr1[i].max(arr2[i]) as u64;
            }

            prev_byte = chunk[7] as i16;
        }

        // Handle remainder with scalar code
        for &byte in remainder {
            let c = BYTE_CLASS_LUT[byte as usize];
            text_count += (c & FLAG_TEXT) as u32;
            high_count += ((c & FLAG_HIGH) >> 1) as u32;
            null_count += ((c & FLAG_NULL) >> 2) as u32;
            let diff = (byte as i16 - prev_byte).abs();
            variation += diff as u64;
            prev_byte = byte as i16;
        }

        let count = data.len() as f32;
        let inv_count = 1.0 / count;

        Self {
            text_ratio: text_count as f32 * inv_count,
            high_ratio: high_count as f32 * inv_count,
            null_ratio: null_count as f32 * inv_count,
            variation: (variation as f32 * inv_count) / 128.0,
        }
    }

    /// Map analysis to RGB color using forensic color scheme.
    ///
    /// Color legend:
    /// - Blue: Nulls / padding / zeroes
    /// - Cyan: ASCII text
    /// - Green: Code / structured data
    /// - Red: High entropy / encrypted data
    #[inline]
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

/// File signature entry for magic byte detection.
struct FileSig {
    magic: &'static [u8],
    name: &'static str,
}

/// Known file signatures sorted by length (longest first for specificity).
static FILE_SIGNATURES: &[FileSig] = &[
    // Longer signatures first for specificity
    FileSig {
        magic: b"SQLite",
        name: "SQLite Database",
    },
    FileSig {
        magic: b"RIFF",
        name: "RIFF Container (WAV/AVI)",
    },
    FileSig {
        magic: &[0x7F, 0x45, 0x4C, 0x46],
        name: "ELF Binary",
    },
    FileSig {
        magic: &[0xCF, 0xFA, 0xED, 0xFE],
        name: "Mach-O Binary",
    },
    FileSig {
        magic: &[0xFE, 0xED, 0xFA, 0xCE],
        name: "Mach-O Binary",
    },
    FileSig {
        magic: &[0xCE, 0xFA, 0xED, 0xFE],
        name: "Mach-O Binary",
    },
    FileSig {
        magic: &[0xCA, 0xFE, 0xBA, 0xBE],
        name: "Mach-O Binary",
    },
    FileSig {
        magic: &[0x25, 0x50, 0x44, 0x46],
        name: "PDF Document",
    },
    FileSig {
        magic: &[0x50, 0x4B, 0x03, 0x04],
        name: "ZIP Archive / Office",
    },
    FileSig {
        magic: &[0x89, 0x50, 0x4E, 0x47],
        name: "PNG Image",
    },
    FileSig {
        magic: &[0x47, 0x49, 0x46, 0x38],
        name: "GIF Image",
    },
    FileSig {
        magic: &[0x52, 0x61, 0x72, 0x21],
        name: "RAR Archive",
    },
    FileSig {
        magic: &[0x37, 0x7A, 0xBC, 0xAF],
        name: "7-Zip Archive",
    },
    FileSig {
        magic: &[0xFF, 0xD8, 0xFF],
        name: "JPEG Image",
    },
    FileSig {
        magic: &[0x42, 0x5A, 0x68],
        name: "BZIP2 Archive",
    },
    FileSig {
        magic: &[0x4D, 0x5A],
        name: "Windows PE (EXE/DLL)",
    },
    FileSig {
        magic: &[0x1F, 0x8B],
        name: "GZIP Archive",
    },
];

/// Identify file type via magic bytes.
/// Optimized with direct signature table lookup.
pub fn identify_file_type(data: &[u8]) -> &'static str {
    if data.len() < 2 {
        return "Unknown Data";
    }

    // Check against signature table
    for sig in FILE_SIGNATURES {
        if data.len() >= sig.magic.len() && data.starts_with(sig.magic) {
            return sig.name;
        }
    }

    // Special case: MP4/MOV (variable header)
    if data.len() > 4
        && data[0] == 0x00
        && data[1] == 0x00
        && data[2] == 0x00
        && (data[4] == 0x66 || data[4] == 0x6D)
    {
        return "MP4/MOV Video";
    }

    "Unknown Binary"
}

/// Fast check if data is likely text (> 80% printable ASCII).
/// Uses SIMD-accelerated 8-way unrolling for counting.
#[inline]
pub fn is_likely_text(data: &[u8]) -> bool {
    if data.is_empty() {
        return false;
    }

    let mut text_count = 0u32;

    // SIMD-accelerated 8-way unrolled counting
    let chunks = data.chunks_exact(8);
    let remainder = chunks.remainder();

    for chunk in chunks {
        // Lookup and accumulate using SIMD
        let c0 = (BYTE_CLASS_LUT[chunk[0] as usize] & FLAG_TEXT) as u32;
        let c1 = (BYTE_CLASS_LUT[chunk[1] as usize] & FLAG_TEXT) as u32;
        let c2 = (BYTE_CLASS_LUT[chunk[2] as usize] & FLAG_TEXT) as u32;
        let c3 = (BYTE_CLASS_LUT[chunk[3] as usize] & FLAG_TEXT) as u32;
        let c4 = (BYTE_CLASS_LUT[chunk[4] as usize] & FLAG_TEXT) as u32;
        let c5 = (BYTE_CLASS_LUT[chunk[5] as usize] & FLAG_TEXT) as u32;
        let c6 = (BYTE_CLASS_LUT[chunk[6] as usize] & FLAG_TEXT) as u32;
        let c7 = (BYTE_CLASS_LUT[chunk[7] as usize] & FLAG_TEXT) as u32;

        // SIMD horizontal add
        let v0 = u32x4::new([c0, c1, c2, c3]);
        let v1 = u32x4::new([c4, c5, c6, c7]);
        let sum0 = v0.to_array();
        let sum1 = v1.to_array();
        text_count += sum0[0] + sum0[1] + sum0[2] + sum0[3] + sum1[0] + sum1[1] + sum1[2] + sum1[3];
    }

    // Handle remainder
    for &byte in remainder {
        text_count += (BYTE_CLASS_LUT[byte as usize] & FLAG_TEXT) as u32;
    }

    // Use integer comparison to avoid float division
    // text_count / len > 0.8  =>  text_count * 10 > len * 8
    text_count * 10 > (data.len() as u32) * 8
}

/// Fast check if data is likely binary/encrypted (high entropy).
/// Uses SIMD-accelerated counting and variation calculation.
#[inline]
pub fn is_likely_encrypted(data: &[u8]) -> bool {
    if data.len() < 32 {
        return false;
    }

    let mut high_count = 0u32;
    let mut variation = 0u64;

    // SIMD-accelerated 8-way unrolled processing
    let chunks = data[1..].chunks_exact(8);
    let remainder = chunks.remainder();
    let mut prev_byte = data[0] as i16;

    for chunk in chunks {
        // Count high bytes
        let h0 = ((BYTE_CLASS_LUT[chunk[0] as usize] & FLAG_HIGH) >> 1) as u32;
        let h1 = ((BYTE_CLASS_LUT[chunk[1] as usize] & FLAG_HIGH) >> 1) as u32;
        let h2 = ((BYTE_CLASS_LUT[chunk[2] as usize] & FLAG_HIGH) >> 1) as u32;
        let h3 = ((BYTE_CLASS_LUT[chunk[3] as usize] & FLAG_HIGH) >> 1) as u32;
        let h4 = ((BYTE_CLASS_LUT[chunk[4] as usize] & FLAG_HIGH) >> 1) as u32;
        let h5 = ((BYTE_CLASS_LUT[chunk[5] as usize] & FLAG_HIGH) >> 1) as u32;
        let h6 = ((BYTE_CLASS_LUT[chunk[6] as usize] & FLAG_HIGH) >> 1) as u32;
        let h7 = ((BYTE_CLASS_LUT[chunk[7] as usize] & FLAG_HIGH) >> 1) as u32;

        high_count += h0 + h1 + h2 + h3 + h4 + h5 + h6 + h7;

        // SIMD variation using i16x8
        let bytes = i16x8::new([
            chunk[0] as i16,
            chunk[1] as i16,
            chunk[2] as i16,
            chunk[3] as i16,
            chunk[4] as i16,
            chunk[5] as i16,
            chunk[6] as i16,
            chunk[7] as i16,
        ]);
        let prev = i16x8::new([
            prev_byte,
            chunk[0] as i16,
            chunk[1] as i16,
            chunk[2] as i16,
            chunk[3] as i16,
            chunk[4] as i16,
            chunk[5] as i16,
            chunk[6] as i16,
        ]);

        let diff1 = bytes - prev;
        let diff2 = prev - bytes;
        let arr1 = diff1.to_array();
        let arr2 = diff2.to_array();

        for i in 0..8 {
            variation += arr1[i].max(arr2[i]) as u64;
        }

        prev_byte = chunk[7] as i16;
    }

    // Handle remainder
    for &byte in remainder {
        high_count += ((BYTE_CLASS_LUT[byte as usize] & FLAG_HIGH) >> 1) as u32;
        variation += (byte as i16 - prev_byte).abs() as u64;
        prev_byte = byte as i16;
    }

    // Use integer arithmetic where possible
    // high_ratio > 0.25  =>  high_count * 4 > len
    // avg_variation > 0.5 * 128 = 64  =>  variation > len * 64
    let len = data.len() as u64;
    high_count as u64 * 4 > len && variation > len * 64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_identify_file_type() {
        assert_eq!(
            identify_file_type(&[0x4D, 0x5A, 0x00, 0x00]),
            "Windows PE (EXE/DLL)"
        );
        assert_eq!(identify_file_type(&[0x7F, 0x45, 0x4C, 0x46]), "ELF Binary");
        assert_eq!(identify_file_type(&[0x89, 0x50, 0x4E, 0x47]), "PNG Image");
    }

    #[test]
    fn test_byte_analysis_nulls() {
        let data = vec![0u8; 100];
        let analysis = ByteAnalysis::analyze(&data);
        assert!(analysis.null_ratio > 0.9);

        let color = analysis.to_color();
        // Should be blue
        assert!(color[2] > color[0]);
        assert!(color[2] > color[1]);
    }

    #[test]
    fn test_byte_analysis_text() {
        let data = b"Hello World! This is ASCII text.";
        let analysis = ByteAnalysis::analyze(data);
        assert!(analysis.text_ratio > 0.9);
    }

    #[test]
    fn test_byte_analysis_high_entropy() {
        // Generate highly varied data (larger range = higher variation)
        let data: Vec<u8> = (0..100).map(|i| ((i * 97 + 13) % 256) as u8).collect();
        let analysis = ByteAnalysis::analyze(&data);
        // Variation is normalized by 128, so typical varied data is 0.2-0.4
        assert!(
            analysis.variation > 0.2,
            "Expected variation > 0.2, got {}",
            analysis.variation
        );
    }

    #[test]
    fn test_lookup_table_correctness() {
        // Verify lookup table values
        assert_eq!(BYTE_CLASS_LUT[0] & FLAG_NULL, FLAG_NULL);
        assert_eq!(BYTE_CLASS_LUT[32] & FLAG_TEXT, FLAG_TEXT);
        assert_eq!(BYTE_CLASS_LUT[126] & FLAG_TEXT, FLAG_TEXT);
        assert_eq!(BYTE_CLASS_LUT[127] & FLAG_TEXT, 0);
        assert_eq!(BYTE_CLASS_LUT[128] & FLAG_HIGH, FLAG_HIGH);
        assert_eq!(BYTE_CLASS_LUT[255] & FLAG_HIGH, FLAG_HIGH);
    }

    #[test]
    fn test_is_likely_text() {
        assert!(is_likely_text(b"Hello World!"));
        assert!(!is_likely_text(&[0xFF, 0xFE, 0x00, 0x01]));
    }
}
