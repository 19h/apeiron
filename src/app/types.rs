//! Core types and data structures for Apeiron application state.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{self, Receiver, SyncSender};
use std::sync::Arc;

use eframe::egui::Vec2;
use memmap2::Mmap;

use crate::hilbert::{d2xy, HilbertBuffer};

// =============================================================================
// Constants
// =============================================================================

/// Interval for sampling Kolmogorov complexity (every N bytes).
pub const COMPLEXITY_SAMPLE_INTERVAL: usize = 64;

/// Interval for sampling RCMSE (every N bytes).
pub const RCMSE_SAMPLE_INTERVAL: usize = 64;

// =============================================================================
// Byte Layout Mode
// =============================================================================

/// How bytes are mapped to 2D space.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum ByteLayout {
    /// Hilbert curve - preserves locality but scrambles linear order.
    #[default]
    Hilbert,
    /// Linear (row-major) - top-left to bottom-right, like reading text.
    /// Better for visualizing sections/segments which are contiguous byte ranges.
    Linear,
}

impl ByteLayout {
    /// Get display name for the layout.
    pub fn name(&self) -> &'static str {
        match self {
            Self::Hilbert => "Hilbert",
            Self::Linear => "Linear",
        }
    }

    /// Get short indicator for HUD.
    pub fn short_name(&self) -> &'static str {
        match self {
            Self::Hilbert => "HIL",
            Self::Linear => "LIN",
        }
    }

    /// Toggle to the other layout.
    pub fn toggle(&self) -> Self {
        match self {
            Self::Hilbert => Self::Linear,
            Self::Linear => Self::Hilbert,
        }
    }
}

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
            | Self::MultiScaleEntropy => file_dimension as f32,
        }
    }

    /// Get the world dimension for this mode, accounting for byte layout.
    /// For modes that support layout toggle, uses the appropriate dimension.
    pub fn world_dimension_for_layout(self, file: &FileData, byte_layout: ByteLayout) -> f32 {
        match self {
            Self::Digraph | Self::BytePhaseSpace => 256.0,
            Self::SimilarityMatrix => file.dimension as f32, // Always uses Hilbert dimension
            Self::Hilbert
            | Self::KolmogorovComplexity
            | Self::JensenShannonDivergence
            | Self::MultiScaleEntropy => file.dimension_for_layout(byte_layout) as f32,
        }
    }

    /// Get the minimum texture size for this mode.
    pub fn min_texture_size(self) -> usize {
        match self {
            // These modes contain exactly 256 bins per axis. Nearest sampling
            // preserves all bins when the native texture is enlarged on screen.
            Self::Digraph | Self::BytePhaseSpace => 256,
            Self::SimilarityMatrix => 2048,
            Self::Hilbert
            | Self::KolmogorovComplexity
            | Self::JensenShannonDivergence
            | Self::MultiScaleEntropy => 512,
        }
    }

    /// Whether this mode should render the full world space.
    pub fn renders_full_world(self) -> bool {
        match self {
            Self::SimilarityMatrix | Self::Digraph | Self::BytePhaseSpace => true,
            Self::Hilbert
            | Self::KolmogorovComplexity
            | Self::JensenShannonDivergence
            | Self::MultiScaleEntropy => false,
        }
    }

    /// Whether this mode supports switching between Hilbert and Linear layout.
    /// Modes with their own coordinate systems (Digraph, PhaseSpace, SimilarityMatrix)
    /// don't support layout switching.
    pub fn supports_layout_toggle(self) -> bool {
        match self {
            Self::Hilbert
            | Self::KolmogorovComplexity
            | Self::JensenShannonDivergence
            | Self::MultiScaleEntropy => true,
            Self::SimilarityMatrix | Self::Digraph | Self::BytePhaseSpace => false,
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
    /// Linear layout dimension (ceil(sqrt(size)) for square layout).
    pub linear_dimension: u64,
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
    /// Progressive Hilbert computation buffer (for large files).
    pub hilbert_buffer: Option<Arc<HilbertBuffer>>,
}

impl FileData {
    /// Get the dimension for the specified byte layout.
    /// - Hilbert: power of 2 dimension for Hilbert curve
    /// - Linear: ceil(sqrt(file_size)) for square layout
    pub fn dimension_for_layout(&self, layout: ByteLayout) -> u64 {
        match layout {
            ByteLayout::Hilbert => self.dimension,
            ByteLayout::Linear => self.linear_dimension,
        }
    }
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

/// Cached outline for the visible hex region.
pub struct OutlineCache {
    /// The range this cache was computed for.
    pub start_offset: u64,
    pub end_offset: u64,
    /// Dimension used to compute the cache.
    pub dimension: u64,
    /// Layout mode used to compute the cache.
    pub byte_layout: ByteLayout,
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
    pub fn is_cache_valid(
        &self,
        start: u64,
        end: u64,
        dimension: u64,
        byte_layout: ByteLayout,
    ) -> bool {
        if let Some(cache) = &self.outline_cache {
            cache.start_offset == start
                && cache.end_offset == end
                && cache.dimension == dimension
                && cache.byte_layout == byte_layout
        } else {
            false
        }
    }

    /// Update the outline cache for the given range.
    pub fn update_outline_cache(
        &mut self,
        start: u64,
        end: u64,
        dimension: u64,
        byte_layout: ByteLayout,
    ) {
        let range_size = end.saturating_sub(start);
        let sample_interval = (range_size / 200).max(1) as usize;

        let mut points = Vec::with_capacity(200);
        let mut min_x = f32::MAX;
        let mut min_y = f32::MAX;
        let mut max_x = f32::MIN;
        let mut max_y = f32::MIN;

        // Helper to convert byte offset to (x, y) based on layout
        let offset_to_xy = |offset: u64| -> (u64, u64) {
            match byte_layout {
                ByteLayout::Hilbert => d2xy(dimension, offset),
                ByteLayout::Linear => (offset % dimension, offset / dimension),
            }
        };

        let mut offset = start;
        while offset < end {
            let (x, y) = offset_to_xy(offset);
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
            let (x, y) = offset_to_xy(end.saturating_sub(1));
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
            byte_layout,
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
    /// Byte layout (Hilbert vs Linear).
    pub byte_layout: ByteLayout,
}

// =============================================================================
// Background Tasks
// =============================================================================

/// Background computation tasks for lazy loading.
pub enum BackgroundTask {
    /// Next complexity sample batch; append to the existing map.
    ComplexityPartial { data: Vec<f32>, progress: f32 },
    /// Final complexity map (computation complete).
    ComplexityComplete,
    /// Next RCMSE sample batch; append to the existing map.
    RcmsePartial { data: Vec<f32>, progress: f32 },
    /// Final RCMSE map (computation complete).
    RcmseComplete,
}

/// Pending background computations.
pub struct BackgroundTasks {
    /// All workers share one bounded channel for this file.
    pub sender: SyncSender<BackgroundTask>,
    /// Cancellation flag shared with workers.
    pub cancelled: Arc<AtomicBool>,
    /// Receiver for completed tasks.
    pub receiver: Receiver<BackgroundTask>,
    /// Whether complexity map is being computed.
    pub computing_complexity: bool,
    /// Whether RCMSE map is being computed.
    pub computing_rcmse: bool,
    /// Progress of complexity map computation (0.0-1.0).
    pub complexity_progress: f32,
    /// Progress of RCMSE map computation (0.0-1.0).
    pub rcmse_progress: f32,
}

impl BackgroundTasks {
    pub fn new() -> Self {
        let (sender, receiver) = mpsc::sync_channel(8);
        Self {
            sender,
            receiver,
            cancelled: Arc::new(AtomicBool::new(false)),
            computing_complexity: false,
            computing_rcmse: false,
            complexity_progress: 0.0,
            rcmse_progress: 0.0,
        }
    }
}

impl Default for BackgroundTasks {
    fn default() -> Self {
        Self::new()
    }
}

impl Drop for BackgroundTasks {
    fn drop(&mut self) {
        self.cancelled.store(true, Ordering::Relaxed);
    }
}
