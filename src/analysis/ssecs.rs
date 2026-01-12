//! SSECS (Suspiciously Structured Entropic Change Score) malware detection.
//!
//! Implementation based on: "Wavelet Decomposition of Software Entropy Reveals
//! Symptoms of Malicious Code" - Wojnowicz et al., Journal of Innovation in
//! Digital Ecosystems 3 (2016) 130-140

use rayon::prelude::*;
use std::collections::HashMap;

use super::wavelet::{compute_entropy_stream, haar_wavelet_transform, wavelet_energy_spectrum};

pub const DEFAULT_CHUNK_SIZE: usize = 256;
pub const MIN_FILE_SIZE: usize = 512;
pub const MAX_LEVELS: usize = 16;

/// Complete wavelet entropy analysis result.
#[derive(Debug, Clone)]
pub struct WaveletEntropyAnalysis {
    pub chunk_size: usize,
    pub entropy_stream: Vec<f64>,
    pub num_chunks: usize,
    pub j: usize,
    pub coefficients: Vec<Vec<f64>>,
    pub energy_spectrum: Vec<f64>,
    pub total_energy: f64,
    pub ssecs: SSECSResult,
}

/// SSECS analysis result.
#[derive(Debug, Clone)]
pub struct SSECSResult {
    pub ssecs_score: f64,
    pub probability_malware: f64,
    pub coarse_energy_ratio: f64,
    pub fine_energy_ratio: f64,
    pub energy_by_level: Vec<(usize, f64)>,
    pub classification: MalwareClassification,
}

/// Malware classification result.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MalwareClassification {
    Clean,
    Suspicious,
    LikelyMalware,
    Unknown,
}

impl SSECSResult {
    /// Map SSECS result to RGB color.
    pub fn to_color(&self) -> [u8; 3] {
        let t = self.probability_malware.clamp(0.0, 1.0);
        match self.classification {
            MalwareClassification::Clean => {
                let s = t * 2.0;
                [
                    (0.1 + s * 0.1) as u8,
                    (0.4 + s * 0.4) as u8,
                    (0.8 - s * 0.3) as u8,
                ]
            }
            MalwareClassification::Suspicious => {
                let s = (t - 0.25) / 0.25;
                [
                    (0.2 + s * 0.3) as u8,
                    (0.8 - s * 0.3) as u8,
                    (0.5 - s * 0.3) as u8,
                ]
            }
            MalwareClassification::LikelyMalware => {
                let s = (t - 0.5) / 0.5;
                [
                    255,
                    ((0.5 - s * 0.5) * 255.0) as u8,
                    ((0.2 + s * 0.3) * 255.0) as u8,
                ]
            }
            MalwareClassification::Unknown => [128, 128, 128],
        }
    }
}

/// Level information for wavelet decomposition.
#[derive(Debug, Clone, Copy)]
pub struct LevelInfo {
    pub level: usize,
    pub num_bins: usize,
    pub bin_size: usize,
    pub energy: f64,
    pub is_coarse: bool,
}

impl WaveletEntropyAnalysis {
    /// Analyze data and compute wavelet entropy metrics.
    pub fn from_data(data: &[u8], chunk_size: usize) -> Self {
        let entropy_stream = compute_entropy_stream(data);
        let num_chunks = entropy_stream.len();

        if num_chunks < 2 {
            return Self {
                chunk_size,
                entropy_stream,
                num_chunks,
                j: 0,
                coefficients: Vec::new(),
                energy_spectrum: Vec::new(),
                total_energy: 0.0,
                ssecs: SSECSResult {
                    ssecs_score: 0.5,
                    probability_malware: 0.5,
                    coarse_energy_ratio: 0.5,
                    fine_energy_ratio: 0.5,
                    energy_by_level: Vec::new(),
                    classification: MalwareClassification::Unknown,
                },
            };
        }

        let j = (num_chunks as f64).log2().floor() as usize;
        let n = 1usize << j;
        let truncated_stream: Vec<f64> = entropy_stream[..n].to_vec();

        let coefficients = haar_wavelet_transform(&truncated_stream);
        let energy_spectrum: Vec<f64> = coefficients
            .iter()
            .map(|coeffs| coeffs.iter().map(|d| d * d).sum())
            .collect();

        let total_energy: f64 = energy_spectrum.iter().sum();
        let ssecs = compute_ssecs(&energy_spectrum, j);

        Self {
            chunk_size,
            entropy_stream,
            num_chunks,
            j,
            coefficients,
            energy_spectrum,
            total_energy,
            ssecs,
        }
    }

    /// Get level information for visualization.
    pub fn get_level_info(&self) -> Vec<LevelInfo> {
        self.energy_spectrum
            .iter()
            .enumerate()
            .map(|(level, &energy)| {
                let num_bins = 1usize << level;
                let bin_size = self.num_chunks / num_bins;
                let is_coarse = level < self.j / 2;
                LevelInfo {
                    level,
                    num_bins,
                    bin_size,
                    energy,
                    is_coarse,
                }
            })
            .collect()
    }
}

/// Compute SSECS score from energy spectrum.
pub fn compute_ssecs(energy_spectrum: &[f64], j: usize) -> SSECSResult {
    let num_levels = energy_spectrum.len();

    if num_levels == 0 {
        return SSECSResult {
            ssecs_score: 0.5,
            probability_malware: 0.5,
            coarse_energy_ratio: 0.5,
            fine_energy_ratio: 0.5,
            energy_by_level: Vec::new(),
            classification: MalwareClassification::Unknown,
        };
    }

    let total_energy: f64 = energy_spectrum.iter().sum();
    let energy_by_level: Vec<(usize, f64)> = energy_spectrum
        .iter()
        .enumerate()
        .map(|(l, &e)| (l, e))
        .collect();

    if total_energy < 1e-10 {
        return SSECSResult {
            ssecs_score: 0.5,
            probability_malware: 0.5,
            coarse_energy_ratio: 0.5,
            fine_energy_ratio: 0.5,
            energy_by_level,
            classification: MalwareClassification::Unknown,
        };
    }

    let coarse_threshold = (j + 1) / 2;
    let coarse_energy: f64 = energy_spectrum[..coarse_threshold].iter().sum();
    let fine_energy: f64 = energy_spectrum[coarse_threshold..].iter().sum();

    let coarse_ratio = coarse_energy / total_energy;
    let fine_ratio = fine_energy / total_energy;

    let probability_malware = logistic_ssecs(coarse_ratio, fine_ratio, energy_spectrum);
    let ssecs_score = probability_malware;

    let classification = if probability_malware < 0.35 {
        MalwareClassification::Clean
    } else if probability_malware < 0.65 {
        MalwareClassification::Suspicious
    } else {
        MalwareClassification::LikelyMalware
    };

    SSECSResult {
        ssecs_score,
        probability_malware,
        coarse_energy_ratio: coarse_ratio,
        fine_energy_ratio: fine_ratio,
        energy_by_level,
        classification,
    }
}

/// Logistic function for SSECS probability.
#[inline]
fn logistic_ssecs(coarse_ratio: f64, fine_ratio: f64, energy_spectrum: &[f64]) -> f64 {
    let beta_coarse = 0.8;
    let beta_fine = -0.6;
    let intercept = -0.2;

    let logit = intercept + beta_coarse * coarse_ratio + beta_fine * fine_ratio;

    let mut weighted_sum = 0.0;
    let mid_level = energy_spectrum.len() / 2;
    for (i, &energy) in energy_spectrum.iter().enumerate().take(mid_level) {
        if i < mid_level / 2 {
            weighted_sum += energy * 0.3;
        } else {
            weighted_sum -= energy * 0.2;
        }
    }

    let adjusted_logit = logit + weighted_sum / 1000.0;

    1.0 / (1.0 + (-adjusted_logit).exp())
}

/// Complete malware analysis report.
#[derive(Debug, Clone)]
pub struct WaveletMalwareReport {
    pub file_size_bytes: usize,
    pub file_size_kb: f64,
    pub file_type: &'static str,
    pub chunk_size: usize,
    pub num_entropy_chunks: usize,
    pub num_wavelet_levels: usize,
    pub total_wavelet_energy: f64,
    pub ssecs: SSECSResult,
    pub level_info: Vec<LevelInfo>,
}

impl WaveletMalwareReport {
    /// Check if the file is suspicious.
    pub fn is_suspicious(&self) -> bool {
        self.ssecs.probability_malware > 0.5
    }

    /// Generate a summary string.
    pub fn summary(&self) -> String {
        format!(
            "File: {} ({} KB, {} chunks)\n\
             Type: {}\n\
             Wavelet Levels: {}\n\
             Total Energy: {:.2}\n\
             SSECS Score: {:.3}\n\
             Malware Probability: {:.1}%\n\
             Classification: {:?}\n\
             Coarse Energy Ratio: {:.2}\n\
             Fine Energy Ratio: {:.2}",
            self.file_type,
            format!("{:.1}", self.file_size_kb),
            self.num_entropy_chunks,
            self.file_type,
            self.num_wavelet_levels,
            self.total_wavelet_energy,
            self.ssecs.ssecs_score,
            self.ssecs.probability_malware * 100.0,
            self.ssecs.classification,
            self.ssecs.coarse_energy_ratio,
            self.ssecs.fine_energy_ratio,
        )
    }
}

/// Analyze a file for malware using wavelet entropy.
pub fn analyze_file_for_malware(data: &[u8]) -> WaveletMalwareReport {
    let analysis = WaveletEntropyAnalysis::from_data(data, DEFAULT_CHUNK_SIZE);

    let file_size_kb = data.len() as f64 / 1024.0;
    let file_type = super::byte::identify_file_type(data);

    let ssecs = analysis.ssecs.clone();
    let level_info = analysis.get_level_info();

    WaveletMalwareReport {
        file_size_bytes: data.len(),
        file_size_kb,
        file_type,
        chunk_size: analysis.chunk_size,
        num_entropy_chunks: analysis.num_chunks,
        num_wavelet_levels: analysis.j,
        total_wavelet_energy: analysis.total_energy,
        ssecs,
        level_info,
    }
}

/// Batch analyze multiple files.
pub fn batch_analyze(files: &[(&[u8], bool)]) -> Vec<(bool, f64)> {
    files
        .par_iter()
        .map(|(data, _is_malware)| {
            let analysis = WaveletEntropyAnalysis::from_data(data, DEFAULT_CHUNK_SIZE);
            let is_malware = analysis.ssecs.probability_malware > 0.5;
            (is_malware, analysis.ssecs.probability_malware)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ssecs_clean_file() {
        let clean_stream: Vec<f64> = vec![
            5.0, 5.1, 5.0, 5.2, 5.1, 5.0, 5.1, 5.0, 5.0, 5.1, 5.0, 5.1, 5.0, 5.1, 5.0, 5.1,
        ];
        let coefficients = haar_wavelet_transform(&clean_stream);
        let energy = wavelet_energy_spectrum(&coefficients);
        let ssecs = compute_ssecs(&energy, 4);

        assert!(
            ssecs.probability_malware < 0.5,
            "Clean file should have low malware probability"
        );
    }

    #[test]
    fn test_ssecs_malicious_pattern() {
        let malicious_stream: Vec<f64> = vec![
            2.0, 2.0, 2.0, 2.0, 7.5, 7.5, 7.5, 7.5, 2.0, 2.0, 2.0, 2.0, 7.5, 7.5, 7.5, 7.5,
        ];
        let coefficients = haar_wavelet_transform(&malicious_stream);
        let energy = wavelet_energy_spectrum(&coefficients);
        let ssecs = compute_ssecs(&energy, 4);

        assert!(
            ssecs.probability_malware > 0.5,
            "Malicious pattern should have high malware probability"
        );
    }

    #[test]
    fn test_batch_analysis() {
        let data1 = vec![0u8; 1024];
        let data2 = vec![0x42u8; 1024];
        let files: Vec<(&[u8], bool)> = vec![(&data1, false), (&data2, false)];

        let results = batch_analyze(&files);
        assert_eq!(results.len(), 2);
    }
}
