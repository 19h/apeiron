//! Byte-level analysis and file type detection.
//!
//! Provides forensic byte analysis and magic byte detection.

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
        // Generate varied data
        let data: Vec<u8> = (0..100).map(|i| ((i * 17 + 31) % 256) as u8).collect();
        let analysis = ByteAnalysis::analyze(&data);
        assert!(analysis.variation > 0.3);
    }
}
