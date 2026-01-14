//! Apeiron - GPU-accelerated binary entropy and complexity visualizer.
//!
//! This library provides tools for analyzing and visualizing binary file entropy
//! and complexity using various algorithms including:
//! - Shannon entropy
//! - Kolmogorov complexity (via tiered XZ/Zstd compression)
//! - Jensen-Shannon divergence
//! - Refined Composite Multi-Scale Entropy (RCMSE)
//! - Wavelet entropy decomposition
//!
//! Visualizations are rendered on Hilbert curves for spatial locality preservation.

pub mod analysis;
pub mod app;
pub mod gpu;
pub mod hilbert;
pub mod util;
pub mod viz;

// Wavelet-based malware detection module
pub mod wavelet_malware;
