//! Apeiron - GPU-accelerated binary entropy and complexity visualizer.
//!
//! This library provides tools for analyzing and visualizing binary file entropy
//! and complexity using various algorithms including:
//! - Shannon entropy
//! - Kolmogorov complexity (via DEFLATE compression)
//! - Jensen-Shannon divergence
//! - Refined Composite Multi-Scale Entropy (RCMSE)
//! - Wavelet-based SSECS malware detection
//!
//! Visualizations are rendered on Hilbert curves for spatial locality preservation.

pub mod analysis;
pub mod app;
pub mod gpu;
pub mod hilbert;
pub mod util;
pub mod viz;

// Legacy module - kept for backwards compatibility during refactoring
pub mod wavelet_malware;
