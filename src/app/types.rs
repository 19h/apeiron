//! Core types and data structures for Apeiron application state.

use std::sync::mpsc::Receiver;
use std::sync::Arc;

use eframe::egui::{TextureHandle, Vec2};
use memmap2::Mmap;

use crate::hilbert::{d2xy, HilbertBuffer};
use crate::wavelet_malware as wm;

// =============================================================================
// Constants
// =============================================================================

/// Interval for sampling Kolmogorov complexity (every N bytes).
pub const COMPLEXITY_SAMPLE_INTERVAL: usize = 64;

/// Interval for sampling RCMSE (every N bytes).
pub const RCMSE_SAMPLE_INTERVAL: usize = 64;

/// Interval for sampling wavelet suspiciousness (every N bytes).
pub const WAVELET_SAMPLE_INTERVAL: usize = 64;

// =============================================================================
// Visualization Mode
// =============================================================================

/// Available visualization modes.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum VisualizationMode {
    /// Hilbert curve with entropy-based coloring (default).
    Hilbert,
    /// Structural similarity matrix (recurrence plot) - reveals repeating patterns.
    SimilarityMatrix,
    /// Byte value digraph - shows byte pair transitions.
    Digraph,
    /// Byte phase space - plots byte[i] vs byte[i+1] showing file trajectory.
    BytePhaseSpace,
    /// Kolmogorov complexity approximation - shows algorithmic complexity via compression.
    KolmogorovComplexity,
    /// Jensen-Shannon divergence - shows distribution anomalies vs file baseline.
    #[default]
    JensenShannonDivergence,
    /// Refined Composite Multi-Scale Entropy - reveals complexity across time scales.
    MultiScaleEntropy,
    /// Wavelet Entropy Decomposition - reveals suspicious entropy patterns via SSECS.
    WaveletEntropy,
}

impl VisualizationMode {
    /// Get display name for the mode.
    pub fn name(&self) -> &'static str {
        match self {
            Self::Hilbert => "Hilbert Curve",
            Self::SimilarityMatrix => "Similarity Matrix",
            Self::Digraph => "Byte Digraph",
            Self::BytePhaseSpace => "Byte Phase Space",
            Self::KolmogorovComplexity => "Kolmogorov Complexity",
            Self::JensenShannonDivergence => "JS Divergence",
            Self::MultiScaleEntropy => "Multi-Scale Entropy (RCMSE)",
            Self::WaveletEntropy => "Wavelet Entropy (SSECS)",
        }
    }

    /// Get all available modes.
    pub fn all() -> &'static [Self] {
        &[
            Self::Hilbert,
            Self::SimilarityMatrix,
            Self::Digraph,
            Self::BytePhaseSpace,
            Self::KolmogorovComplexity,
            Self::JensenShannonDivergence,
            Self::MultiScaleEntropy,
            Self::WaveletEntropy,
        ]
    }

    /// Get the world dimension for this mode.
    /// Digraph and BytePhaseSpace use a fixed 256×256 space (byte values).
    /// Other modes scale with file size.
    pub fn world_dimension(self, file_dimension: u64) -> f32 {
        match self {
            Self::Digraph | Self::BytePhaseSpace => 256.0,
            Self::Hilbert
            | Self::SimilarityMatrix
            | Self::KolmogorovComplexity
            | Self::JensenShannonDivergence
            | Self::MultiScaleEntropy
            | Self::WaveletEntropy => file_dimension as f32,
        }
    }

    /// Get the minimum texture size for this mode.
    pub fn min_texture_size(self) -> usize {
        match self {
            Self::Digraph | Self::BytePhaseSpace => 2048,
            Self::SimilarityMatrix => 2048,
            Self::Hilbert
            | Self::KolmogorovComplexity
            | Self::JensenShannonDivergence
            | Self::MultiScaleEntropy
            | Self::WaveletEntropy => 512,
        }
    }

    /// Whether this mode should render the full world space.
    pub fn renders_full_world(self) -> bool {
        match self {
            Self::SimilarityMatrix | Self::Digraph | Self::BytePhaseSpace => true,
            Self::Hilbert
            | Self::KolmogorovComplexity
            | Self::JensenShannonDivergence
            | Self::MultiScaleEntropy
            | Self::WaveletEntropy => false,
        }
    }
}

// =============================================================================
// File Data
// =============================================================================

/// Loaded file information and data.
pub struct FileData {
    /// Memory-mapped file data (efficient for large files).
    pub data: Arc<Mmap>,
    /// File size in bytes.
    pub size: u64,
    /// Hilbert curve dimension (power of 2).
    pub dimension: u64,
    /// Detected file type.
    pub file_type: &'static str,
    /// Original file path.
    #[allow(dead_code)]
    pub path: std::path::PathBuf,
    /// Precomputed Kolmogorov complexity values.
    pub complexity_map: Option<Arc<Vec<f32>>>,
    /// Reference byte distribution for JSD calculation.
    pub reference_distribution: Arc<[f64; 256]>,
    /// Precomputed RCMSE values.
    pub rcmse_map: Option<Arc<Vec<f32>>>,
    /// Precomputed wavelet suspiciousness values.
    pub wavelet_map: Option<Arc<Vec<f32>>>,
    /// Progressive Hilbert computation buffer (for large files).
    pub hilbert_buffer: Option<Arc<HilbertBuffer>>,
}

// =============================================================================
// Viewport
// =============================================================================

/// Viewport state for pan and zoom.
#[derive(Clone, Copy)]
pub struct Viewport {
    /// Zoom level (1.0 = 100%).
    pub zoom: f32,
    /// Pan offset in world coordinates.
    pub offset: Vec2,
}

impl Default for Viewport {
    fn default() -> Self {
        Self {
            zoom: 1.0,
            offset: Vec2::ZERO,
        }
    }
}

// =============================================================================
// Selection
// =============================================================================

/// Selected/hovered byte information.
#[derive(Default)]
pub struct Selection {
    /// Byte offset in the file.
    pub offset: u64,
    /// Shannon entropy of the selected window.
    pub entropy: f64,
    /// Kolmogorov complexity approximation.
    pub kolmogorov_complexity: f64,
    /// Extracted ASCII string from the selected window.
    pub ascii_string: Option<String>,
}

// =============================================================================
// Hex View
// =============================================================================

/// Cached Hilbert curve outline for the visible hex region.
pub struct OutlineCache {
    /// The range this cache was computed for.
    pub start_offset: u64,
    pub end_offset: u64,
    /// Dimension used to compute the cache.
    pub dimension: u64,
    /// Precomputed world coordinates (x, y) for the outline.
    pub points: Vec<(f32, f32)>,
    /// Bounding box in world coordinates.
    pub bbox: (f32, f32, f32, f32), // min_x, min_y, max_x, max_y
}

/// Hex view state for the interactive hex panel.
pub struct HexView {
    /// Starting byte offset of the visible hex view.
    pub scroll_offset: u64,
    /// Number of visible rows in the hex view.
    pub visible_rows: usize,
    /// Bytes per row (typically 16).
    pub bytes_per_row: usize,
    /// Currently hovered row in hex view.
    #[allow(dead_code)]
    pub hovered_row: Option<usize>,
    /// Cached outline data for the visible region.
    pub outline_cache: Option<OutlineCache>,
}

impl Default for HexView {
    fn default() -> Self {
        Self::new()
    }
}

impl HexView {
    pub fn new() -> Self {
        Self {
            scroll_offset: 0,
            visible_rows: 32,
            bytes_per_row: 16,
            hovered_row: None,
            outline_cache: None,
        }
    }

    /// Get the byte range currently visible in the hex view.
    pub fn visible_range(&self) -> (u64, u64) {
        let start = self.scroll_offset;
        let end = start + (self.visible_rows * self.bytes_per_row) as u64;
        (start, end)
    }

    /// Scroll to center the given offset in the hex view.
    pub fn scroll_to(&mut self, offset: u64, file_size: u64) {
        let row = offset / self.bytes_per_row as u64;
        let center_row = self.visible_rows as u64 / 2;
        let target_row = row.saturating_sub(center_row);
        let max_row = file_size.saturating_sub(1) / self.bytes_per_row as u64;
        let max_start_row = max_row.saturating_sub(self.visible_rows as u64 - 1);
        self.scroll_offset = target_row.min(max_start_row) * self.bytes_per_row as u64;
    }

    /// Check if the outline cache is valid for the current visible range.
    pub fn is_cache_valid(&self, start: u64, end: u64, dimension: u64) -> bool {
        if let Some(cache) = &self.outline_cache {
            cache.start_offset == start && cache.end_offset == end && cache.dimension == dimension
        } else {
            false
        }
    }

    /// Update the outline cache for the given range.
    pub fn update_outline_cache(&mut self, start: u64, end: u64, dimension: u64) {
        let range_size = end.saturating_sub(start);
        let sample_interval = (range_size / 200).max(1) as usize;

        let mut points = Vec::with_capacity(200);
        let mut min_x = f32::MAX;
        let mut min_y = f32::MAX;
        let mut max_x = f32::MIN;
        let mut max_y = f32::MIN;

        let mut offset = start;
        while offset < end {
            let (x, y) = d2xy(dimension, offset);
            let fx = x as f32 + 0.5;
            let fy = y as f32 + 0.5;
            points.push((fx, fy));

            min_x = min_x.min(x as f32);
            min_y = min_y.min(y as f32);
            max_x = max_x.max(x as f32 + 1.0);
            max_y = max_y.max(y as f32 + 1.0);

            offset += sample_interval as u64;
        }

        // Add final point
        if end > start {
            let (x, y) = d2xy(dimension, end.saturating_sub(1));
            points.push((x as f32 + 0.5, y as f32 + 0.5));
            min_x = min_x.min(x as f32);
            min_y = min_y.min(y as f32);
            max_x = max_x.max(x as f32 + 1.0);
            max_y = max_y.max(y as f32 + 1.0);
        }

        self.outline_cache = Some(OutlineCache {
            start_offset: start,
            end_offset: end,
            dimension,
            points,
            bbox: (min_x, min_y, max_x, max_y),
        });
    }
}

// =============================================================================
// Texture Parameters
// =============================================================================

/// Parameters used to generate the current texture.
#[derive(Clone, Copy, PartialEq)]
pub struct TextureParams {
    /// World-space region covered by texture.
    pub world_min: Vec2,
    pub world_max: Vec2,
    /// Texture resolution.
    pub tex_size: usize,
    /// Visualization mode used to generate this texture.
    pub viz_mode: VisualizationMode,
}

// =============================================================================
// Background Tasks
// =============================================================================

/// Background computation tasks for lazy loading.
pub enum BackgroundTask {
    /// Partial complexity map update (streaming).
    ComplexityPartial { data: Vec<f32>, progress: f32 },
    /// Final complexity map (computation complete).
    ComplexityComplete,
    /// Partial RCMSE map update (streaming).
    RcmsePartial { data: Vec<f32>, progress: f32 },
    /// Final RCMSE map (computation complete).
    RcmseComplete,
    /// Partial wavelet map update (streaming).
    WaveletPartial { data: Vec<f32>, progress: f32 },
    /// Final wavelet map (computation complete).
    WaveletComplete,
    /// Wavelet report (not streamed, computed once).
    WaveletReport(wm::WaveletMalwareReport),
}

/// Pending background computations.
pub struct BackgroundTasks {
    /// Receiver for completed tasks.
    pub receiver: Receiver<BackgroundTask>,
    /// Whether complexity map is being computed.
    pub computing_complexity: bool,
    /// Whether RCMSE map is being computed.
    pub computing_rcmse: bool,
    /// Whether wavelet map is being computed.
    pub computing_wavelet: bool,
    /// Whether wavelet report is being computed.
    pub computing_wavelet_report: bool,
    /// Progress of complexity map computation (0.0-1.0).
    pub complexity_progress: f32,
    /// Progress of RCMSE map computation (0.0-1.0).
    pub rcmse_progress: f32,
    /// Progress of wavelet map computation (0.0-1.0).
    pub wavelet_progress: f32,
}
