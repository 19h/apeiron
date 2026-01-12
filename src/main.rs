//! Apeiron - Binary file entropy and complexity visualizer using Hilbert curves.
//!
//! Apeiron - GPU-accelerated binary entropy visualizer using Hilbert curves
//! through entropy-based color mapping on a Hilbert curve layout.

#![warn(clippy::all, clippy::pedantic)]
#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]

mod analysis;
mod gpu;
mod hilbert;
mod hilbert_refiner;
mod wavelet_malware;

use std::fs::File;
use std::path::PathBuf;
use std::sync::mpsc::{self, Receiver, TryRecvError};
use std::sync::Arc;
use std::thread;
use std::time::Duration;

use memmap2::Mmap;

use eframe::egui::{self, Color32, ColorImage, Pos2, Rect, RichText, Sense, TextureHandle, Vec2};
use rayon::prelude::*;

use analysis::{
    byte_distribution_sampled, calculate_entropy, calculate_jsd, calculate_kolmogorov_complexity,
    calculate_rcmse_quick, extract_ascii, format_bytes, identify_file_type,
    precompute_chunk_entropies, wavelet_suspiciousness_8, ByteAnalysis, JSDAnalysis,
    KolmogorovAnalysis, ANALYSIS_WINDOW, WAVELET_CHUNK_SIZE,
};
use hilbert::{calculate_dimension, d2xy, xy2d};
use hilbert_refiner::{HilbertBuffer, HilbertRefiner, PROGRESSIVE_THRESHOLD};
use wavelet_malware as wm;

// =============================================================================
// Application State
// =============================================================================

/// Available visualization modes.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
enum VisualizationMode {
    /// Hilbert curve with entropy-based coloring (default).
    #[default]
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
    JensenShannonDivergence,
    /// Refined Composite Multi-Scale Entropy - reveals complexity across time scales.
    MultiScaleEntropy,
    /// Wavelet Entropy Decomposition - reveals suspicious entropy patterns via SSECS.
    WaveletEntropy,
}

impl VisualizationMode {
    /// Get display name for the mode.
    fn name(&self) -> &'static str {
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
    fn all() -> &'static [Self] {
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
    fn world_dimension(self, file_dimension: u64) -> f32 {
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
    /// Higher values give better quality but use more memory.
    fn min_texture_size(self) -> usize {
        match self {
            // Digraph/Phase are 256×256 grids - use 2048 for 8x supersampling
            Self::Digraph | Self::BytePhaseSpace => 2048,
            // Similarity matrix benefits from high consistent resolution
            Self::SimilarityMatrix => 2048,
            // Hilbert can use smaller textures when zoomed out
            Self::Hilbert => 512,
            // Kolmogorov, JSD, RCMSE, and Wavelet use Hilbert mapping, same requirements
            Self::KolmogorovComplexity
            | Self::JensenShannonDivergence
            | Self::MultiScaleEntropy
            | Self::WaveletEntropy => 512,
        }
    }

    /// Whether this mode should render the full world space (ignore viewport for texture generation).
    /// The texture covers the entire visualization, and viewport only affects which part is displayed.
    /// This ensures consistent sampling at the cost of less detail when zoomed in.
    fn renders_full_world(self) -> bool {
        match self {
            // SimilarityMatrix needs consistent sampling - render full matrix
            Self::SimilarityMatrix => true,
            // Digraph and BytePhaseSpace are fixed 256×256, render full grid
            Self::Digraph | Self::BytePhaseSpace => true,
            // Hilbert, Kolmogorov, JSD, RCMSE, and Wavelet benefit from viewport-aware rendering
            Self::Hilbert
            | Self::KolmogorovComplexity
            | Self::JensenShannonDivergence
            | Self::MultiScaleEntropy
            | Self::WaveletEntropy => false,
        }
    }
}

/// A single tab containing an open file and its state.
struct Tab {
    /// Tab title (file name).
    title: String,
    /// Loaded file data.
    file_data: Option<FileData>,
    /// Viewport state for pan/zoom.
    viewport: Viewport,
    /// Currently selected/hovered byte information.
    selection: Selection,
    /// Hex view state.
    hex_view: HexView,
    /// Cached wavelet malware analysis for the current file.
    wavelet_report: Option<wavelet_malware::WaveletMalwareReport>,
    /// Cached texture for the entropy visualization.
    texture: Option<TextureHandle>,
    /// Texture generation parameters (to detect when regeneration is needed).
    texture_params: Option<TextureParams>,
    /// Current visualization mode.
    viz_mode: VisualizationMode,
    /// Whether viewport needs to be fitted to view (after load or mode change).
    needs_fit_to_view: bool,
    /// View size used for the last fit_to_view call (to detect when re-fit is needed).
    last_fit_view_size: Option<Vec2>,
    /// Background computation tasks.
    background_tasks: Option<BackgroundTasks>,
}

impl Default for Tab {
    fn default() -> Self {
        Self {
            title: "New Tab".to_string(),
            file_data: None,
            viewport: Viewport::default(),
            selection: Selection::default(),
            hex_view: HexView::new(),
            wavelet_report: None,
            texture: None,
            texture_params: None,
            viz_mode: VisualizationMode::default(),
            needs_fit_to_view: false,
            last_fit_view_size: None,
            background_tasks: None,
        }
    }
}

/// Main application state.
struct ApeironApp {
    /// Open tabs.
    tabs: Vec<Tab>,
    /// Index of the currently active tab.
    active_tab: usize,
    /// Whether the help popup is shown.
    show_help: bool,
    /// Whether a file is being dragged over.
    is_drop_target: bool,
    /// GPU renderer for accelerated visualization.
    gpu_renderer: Option<gpu::GpuRenderer>,
    /// Initial file to load (from command-line argument).
    initial_file: Option<PathBuf>,
    /// Tab currently being dragged (index).
    dragging_tab: Option<usize>,
}

/// Parameters used to generate the current texture.
#[derive(Clone, Copy, PartialEq)]
struct TextureParams {
    /// World-space region covered by texture.
    world_min: Vec2,
    world_max: Vec2,
    /// Texture resolution.
    tex_size: usize,
    /// Visualization mode used to generate this texture.
    viz_mode: VisualizationMode,
}

/// Loaded file information and data.
struct FileData {
    /// Memory-mapped file data (efficient for large files).
    data: Arc<Mmap>,
    /// File size in bytes.
    size: u64,
    /// Hilbert curve dimension (power of 2).
    dimension: u64,
    /// Detected file type.
    file_type: &'static str,
    /// Original file path.
    #[allow(dead_code)]
    path: PathBuf,
    /// Precomputed Kolmogorov complexity values (one per COMPLEXITY_SAMPLE_INTERVAL bytes).
    /// None if not yet computed.
    complexity_map: Option<Arc<Vec<f32>>>,
    /// Reference byte distribution for JSD calculation (whole file distribution).
    reference_distribution: Arc<[f64; 256]>,
    /// Precomputed RCMSE values (one per RCMSE_SAMPLE_INTERVAL bytes).
    /// None if not yet computed.
    rcmse_map: Option<Arc<Vec<f32>>>,
    /// Precomputed wavelet suspiciousness values (one per WAVELET_SAMPLE_INTERVAL bytes).
    /// None if not yet computed.
    wavelet_map: Option<Arc<Vec<f32>>>,
    /// Progressive Hilbert computation buffer (for large files).
    /// None for small files that compute immediately.
    hilbert_buffer: Option<Arc<HilbertBuffer>>,
}

/// Background computation tasks for lazy loading.
enum BackgroundTask {
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
struct BackgroundTasks {
    /// Receiver for completed tasks.
    receiver: Receiver<BackgroundTask>,
    /// Whether complexity map is being computed.
    computing_complexity: bool,
    /// Whether RCMSE map is being computed.
    computing_rcmse: bool,
    /// Whether wavelet map is being computed.
    computing_wavelet: bool,
    /// Whether wavelet report is being computed.
    computing_wavelet_report: bool,
    /// Progress of complexity map computation (0.0-1.0).
    complexity_progress: f32,
    /// Progress of RCMSE map computation (0.0-1.0).
    rcmse_progress: f32,
    /// Progress of wavelet map computation (0.0-1.0).
    wavelet_progress: f32,
}

/// Interval for sampling RCMSE (every N bytes).
/// Same as complexity for consistent detail level.
const RCMSE_SAMPLE_INTERVAL: usize = 64;

/// Interval for sampling Kolmogorov complexity (every N bytes).
/// Smaller = more precise but slower to precompute.
const COMPLEXITY_SAMPLE_INTERVAL: usize = 64;

/// Interval for sampling wavelet suspiciousness (every N bytes).
/// Same as complexity/RCMSE for consistent detail level.
const WAVELET_SAMPLE_INTERVAL: usize = 64;

/// Viewport state for pan and zoom.
#[derive(Clone, Copy)]
struct Viewport {
    /// Zoom level (1.0 = 100%).
    zoom: f32,
    /// Pan offset in world coordinates.
    offset: Vec2,
}

impl Default for Viewport {
    fn default() -> Self {
        Self {
            zoom: 1.0,
            offset: Vec2::ZERO,
        }
    }
}

/// Selected/hovered byte information.
#[derive(Default)]
struct Selection {
    /// Byte offset in the file.
    offset: u64,
    /// Shannon entropy of the selected window.
    entropy: f64,
    /// Kolmogorov complexity approximation (compression ratio) of the selected window.
    kolmogorov_complexity: f64,
    /// Extracted ASCII string from the selected window (if any).
    ascii_string: Option<String>,
}

/// Hex view state for the interactive hex panel.
struct HexView {
    /// Starting byte offset of the visible hex view.
    scroll_offset: u64,
    /// Number of visible rows in the hex view.
    visible_rows: usize,
    /// Bytes per row (typically 16).
    bytes_per_row: usize,
    /// Currently hovered row in hex view (for highlighting).
    /// Reserved for future row hover highlighting feature.
    #[allow(dead_code)]
    hovered_row: Option<usize>,
    /// Cached outline data for the visible region.
    outline_cache: Option<OutlineCache>,
}

/// Cached Hilbert curve outline for the visible hex region.
struct OutlineCache {
    /// The range this cache was computed for.
    start_offset: u64,
    end_offset: u64,
    /// Dimension used to compute the cache.
    dimension: u64,
    /// Precomputed world coordinates (x, y) for the outline.
    points: Vec<(f32, f32)>,
    /// Bounding box in world coordinates.
    bbox: (f32, f32, f32, f32), // min_x, min_y, max_x, max_y
}

impl Default for HexView {
    fn default() -> Self {
        Self::new()
    }
}

impl HexView {
    fn new() -> Self {
        Self {
            scroll_offset: 0,
            visible_rows: 32,
            bytes_per_row: 16,
            hovered_row: None,
            outline_cache: None,
        }
    }

    /// Get the byte range currently visible in the hex view.
    fn visible_range(&self) -> (u64, u64) {
        let start = self.scroll_offset;
        let end = start + (self.visible_rows * self.bytes_per_row) as u64;
        (start, end)
    }

    /// Scroll to center the given offset in the hex view.
    fn scroll_to(&mut self, offset: u64, file_size: u64) {
        let row = offset / self.bytes_per_row as u64;
        let center_row = self.visible_rows as u64 / 2;
        let target_row = row.saturating_sub(center_row);
        let max_row = file_size.saturating_sub(1) / self.bytes_per_row as u64;
        let max_start_row = max_row.saturating_sub(self.visible_rows as u64 - 1);
        self.scroll_offset = target_row.min(max_start_row) * self.bytes_per_row as u64;
    }

    /// Check if the outline cache is valid for the current visible range.
    fn is_cache_valid(&self, start: u64, end: u64, dimension: u64) -> bool {
        if let Some(cache) = &self.outline_cache {
            cache.start_offset == start && cache.end_offset == end && cache.dimension == dimension
        } else {
            false
        }
    }

    /// Update the outline cache for the given range.
    fn update_outline_cache(&mut self, start: u64, end: u64, dimension: u64) {
        // Sample at reasonable intervals - fewer points for performance
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
// Application Implementation
// =============================================================================

impl Default for ApeironApp {
    fn default() -> Self {
        Self {
            tabs: vec![Tab::default()],
            active_tab: 0,
            show_help: false,
            is_drop_target: false,
            gpu_renderer: None,
            initial_file: None,
            dragging_tab: None,
        }
    }
}

impl ApeironApp {
    /// Create a new application instance.
    #[allow(dead_code)]
    fn new(cc: &eframe::CreationContext<'_>) -> Self {
        Self::new_with_file(cc, None)
    }

    /// Create a new application instance with an optional initial file to load.
    fn new_with_file(_cc: &eframe::CreationContext<'_>, initial_file: Option<PathBuf>) -> Self {
        let gpu_renderer = gpu::GpuRenderer::new();
        if gpu_renderer.is_some() {
            println!("GPU acceleration enabled");
        } else {
            println!("GPU acceleration unavailable, using CPU fallback");
        }
        Self {
            gpu_renderer,
            initial_file,
            ..Self::default()
        }
    }

    /// Get a reference to the active tab.
    fn active_tab(&self) -> &Tab {
        &self.tabs[self.active_tab]
    }

    /// Get a mutable reference to the active tab.
    fn active_tab_mut(&mut self) -> &mut Tab {
        &mut self.tabs[self.active_tab]
    }

    /// Check if any tab has a file loaded.
    #[allow(dead_code)]
    fn has_any_file(&self) -> bool {
        self.tabs.iter().any(|t| t.file_data.is_some())
    }

    /// Open a new tab and make it active.
    fn new_tab(&mut self) {
        self.tabs.push(Tab::default());
        self.active_tab = self.tabs.len() - 1;
    }

    /// Close the tab at the given index.
    fn close_tab(&mut self, index: usize) {
        if self.tabs.len() <= 1 {
            // Don't close the last tab, just clear it
            self.tabs[0] = Tab::default();
            self.active_tab = 0;
            return;
        }

        self.tabs.remove(index);

        // Adjust active tab if needed
        if self.active_tab >= self.tabs.len() {
            self.active_tab = self.tabs.len() - 1;
        } else if self.active_tab > index {
            self.active_tab -= 1;
        }
    }

    /// Maximum recommended file size for analysis.
    /// Files larger than this will show a warning about long computation times.
    const LARGE_FILE_WARNING_SIZE: u64 = 10 * 1024 * 1024 * 1024; // 10GB

    /// Load a file from the given path.
    /// If `in_new_tab` is true, opens in a new tab; otherwise loads in the current tab.
    /// Uses memory-mapped files to efficiently handle large files without loading entirely into RAM.
    fn load_file(&mut self, path: PathBuf, in_new_tab: bool) {
        // Open file and memory-map it
        let file = match File::open(&path) {
            Ok(f) => f,
            Err(e) => {
                eprintln!("Error opening file: {e}");
                return;
            }
        };

        let mmap = match unsafe { Mmap::map(&file) } {
            Ok(m) => m,
            Err(e) => {
                eprintln!("Error memory-mapping file: {e}");
                return;
            }
        };

        let data = &mmap[..];
        let size = data.len() as u64;

        if size == 0 {
            eprintln!("Error: File is empty");
            return;
        }

        // Warn about very large files
        if size > Self::LARGE_FILE_WARNING_SIZE {
            println!(
                "Warning: File is very large ({:.1} GB). Background analysis may take a while.",
                size as f64 / (1024.0 * 1024.0 * 1024.0)
            );
        }

        let dimension = calculate_dimension(size);
        let file_type = identify_file_type(data);

        // Extract filename for tab title
        let title = path
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("Unknown")
            .to_string();

        // Upload to GPU if available (has its own size limit)
        if let Some(ref mut gpu) = self.gpu_renderer {
            gpu.upload_file(data, dimension);
        }

        // Calculate reference byte distribution for JSD using sampling for large files
        // This is O(1) for large files instead of O(n)
        let reference_distribution = Arc::new(byte_distribution_sampled(data));

        let data = Arc::new(mmap);

        // Create or switch to the target tab
        if in_new_tab {
            self.new_tab();
        }
        let tab = self.active_tab_mut();

        // Start progressive Hilbert computation for large files
        let hilbert_buffer = if size > PROGRESSIVE_THRESHOLD {
            println!(
                "File > {}MB: Using progressive Hilbert rendering",
                PROGRESSIVE_THRESHOLD / (1024 * 1024)
            );
            Some(HilbertRefiner::start(Arc::clone(&data), size))
        } else {
            None
        };

        // Create file data with empty precomputed maps
        tab.file_data = Some(FileData {
            data: Arc::clone(&data),
            size,
            dimension,
            file_type,
            path,
            complexity_map: None,
            reference_distribution,
            rcmse_map: None,
            wavelet_map: None,
            hilbert_buffer,
        });

        // Set tab title
        tab.title = title;

        // Clear previous wavelet report
        tab.wavelet_report = None;

        // LAZY COMPUTATION: Only spawn WaveletReport on load (needed for sidebar).
        // Other mode-specific computations are started when user switches to that mode.
        let (tx, rx) = mpsc::channel();
        let data_for_report = Arc::clone(&data);
        thread::spawn(move || {
            let report = wm::analyze_file_for_malware(&data_for_report);
            let _ = tx.send(BackgroundTask::WaveletReport(report));
        });

        let tab = self.active_tab_mut();
        tab.background_tasks = Some(BackgroundTasks {
            receiver: rx,
            computing_complexity: false, // Lazy: not started yet
            computing_rcmse: false,      // Lazy: not started yet
            computing_wavelet: false,    // Lazy: not started yet
            computing_wavelet_report: true,
            complexity_progress: 0.0,
            rcmse_progress: 0.0,
            wavelet_progress: 0.0,
        });

        // Reset viewport and selection
        tab.viewport = Viewport::default();
        tab.texture = None;
        tab.texture_params = None;
        tab.needs_fit_to_view = true;

        // Initialize selection at offset 0
        Self::update_selection_on_tab(tab, 0);

        println!(
            "Loaded file: Size={} bytes, Dimension={}, Type={}",
            size, dimension, file_type
        );
        if size > PROGRESSIVE_THRESHOLD {
            println!("Progressive Hilbert rendering started...");
        }
        println!("Wavelet analysis started (other modes computed on demand)...");
    }

    /// Poll for completed background tasks on the active tab and update file data.
    /// Returns true if any task completed or updated (may need texture refresh).
    fn poll_background_tasks(&mut self) -> bool {
        let tab = &mut self.tabs[self.active_tab];
        let Some(tasks) = &mut tab.background_tasks else {
            return false;
        };

        let mut any_updated = false;

        // Poll all available messages (partial updates and completions)
        loop {
            match tasks.receiver.try_recv() {
                Ok(task) => {
                    any_updated = true;
                    match task {
                        BackgroundTask::ComplexityPartial { data, progress } => {
                            if let Some(ref mut file) = tab.file_data {
                                file.complexity_map = Some(Arc::new(data));
                            }
                            tasks.complexity_progress = progress;
                        }
                        BackgroundTask::ComplexityComplete => {
                            println!(
                                "Complexity map complete: {} samples",
                                tab.file_data
                                    .as_ref()
                                    .and_then(|f| f.complexity_map.as_ref())
                                    .map(|m| m.len())
                                    .unwrap_or(0)
                            );
                            tasks.computing_complexity = false;
                            tasks.complexity_progress = 1.0;
                        }
                        BackgroundTask::RcmsePartial { data, progress } => {
                            if let Some(ref mut file) = tab.file_data {
                                file.rcmse_map = Some(Arc::new(data));
                            }
                            tasks.rcmse_progress = progress;
                        }
                        BackgroundTask::RcmseComplete => {
                            println!(
                                "RCMSE map complete: {} samples",
                                tab.file_data
                                    .as_ref()
                                    .and_then(|f| f.rcmse_map.as_ref())
                                    .map(|m| m.len())
                                    .unwrap_or(0)
                            );
                            tasks.computing_rcmse = false;
                            tasks.rcmse_progress = 1.0;
                        }
                        BackgroundTask::WaveletPartial { data, progress } => {
                            if let Some(ref mut file) = tab.file_data {
                                file.wavelet_map = Some(Arc::new(data));
                            }
                            tasks.wavelet_progress = progress;
                        }
                        BackgroundTask::WaveletComplete => {
                            println!(
                                "Wavelet map complete: {} samples",
                                tab.file_data
                                    .as_ref()
                                    .and_then(|f| f.wavelet_map.as_ref())
                                    .map(|m| m.len())
                                    .unwrap_or(0)
                            );
                            tasks.computing_wavelet = false;
                            tasks.wavelet_progress = 1.0;
                        }
                        BackgroundTask::WaveletReport(report) => {
                            println!(
                                "Wavelet analysis ready: SSECS={:.3}, Malware Prob={:.1}%",
                                report.ssecs.ssecs_score,
                                report.ssecs.probability_malware * 100.0
                            );
                            tab.wavelet_report = Some(report);
                            tasks.computing_wavelet_report = false;
                        }
                    }
                }
                Err(TryRecvError::Empty) => break,
                Err(TryRecvError::Disconnected) => {
                    // All senders dropped, no more tasks coming
                    tab.background_tasks = None;
                    break;
                }
            }
        }

        // Clear background tasks if all done
        let tab = &mut self.tabs[self.active_tab];
        if let Some(tasks) = &tab.background_tasks {
            if !tasks.computing_complexity
                && !tasks.computing_rcmse
                && !tasks.computing_wavelet
                && !tasks.computing_wavelet_report
            {
                self.tabs[self.active_tab].background_tasks = None;
                println!("All background computations complete.");
            }
        }

        any_updated
    }

    /// Check if a specific visualization mode's data is ready for the active tab.
    fn is_mode_data_ready(&self, mode: VisualizationMode) -> bool {
        let tab = self.active_tab();
        let Some(ref file) = tab.file_data else {
            return false;
        };

        match mode {
            // These modes don't need precomputed maps
            VisualizationMode::Hilbert
            | VisualizationMode::SimilarityMatrix
            | VisualizationMode::Digraph
            | VisualizationMode::BytePhaseSpace => true,
            // JSD needs reference distribution (always computed eagerly)
            VisualizationMode::JensenShannonDivergence => true,
            // These need precomputed maps
            VisualizationMode::KolmogorovComplexity => file.complexity_map.is_some(),
            VisualizationMode::MultiScaleEntropy => file.rcmse_map.is_some(),
            VisualizationMode::WaveletEntropy => file.wavelet_map.is_some(),
        }
    }

    /// Check if any background computation is in progress for the active tab.
    fn is_computing(&self) -> bool {
        let tab = self.active_tab();
        // Check if standard background tasks are running
        if tab.background_tasks.is_some() {
            return true;
        }
        // Check if Hilbert refinement is in progress (and not paused)
        if let Some(ref file) = tab.file_data {
            if let Some(ref buffer) = file.hilbert_buffer {
                if !buffer.is_fine_complete() && !buffer.is_paused() {
                    return true;
                }
            }
        }
        false
    }

    /// Check if Hilbert refinement is in progress and we're in Hilbert mode.
    /// Used to determine if we need continuous repaint for progressive rendering.
    fn is_hilbert_refining(&self) -> bool {
        let tab = self.active_tab();
        if tab.viz_mode != VisualizationMode::Hilbert {
            return false;
        }
        if let Some(ref file) = tab.file_data {
            if let Some(ref buffer) = file.hilbert_buffer {
                return !buffer.is_fine_complete();
            }
        }
        false
    }

    /// Get Hilbert refinement progress (0.0 to 1.0) if active.
    /// Returns None if no hilbert buffer or if refinement is complete.
    fn get_hilbert_refine_progress(&self) -> Option<f32> {
        let tab = self.active_tab();
        if let Some(ref file) = tab.file_data {
            if let Some(ref buffer) = file.hilbert_buffer {
                if !buffer.is_fine_complete() {
                    return Some(buffer.progress_fraction());
                }
            }
        }
        None
    }

    /// Look up precomputed complexity value for a file offset.
    fn lookup_complexity(complexity_map: &[f32], offset: usize) -> f32 {
        if complexity_map.is_empty() {
            return 0.0;
        }
        let index = offset / COMPLEXITY_SAMPLE_INTERVAL;
        complexity_map.get(index).copied().unwrap_or(0.0)
    }

    /// Look up precomputed RCMSE value for a file offset.
    fn lookup_rcmse(rcmse_map: &[f32], offset: usize) -> f32 {
        if rcmse_map.is_empty() {
            return 0.5;
        }
        let index = offset / RCMSE_SAMPLE_INTERVAL;
        rcmse_map.get(index).copied().unwrap_or(0.5)
    }

    /// Look up precomputed wavelet suspiciousness value for a file offset.
    fn lookup_wavelet(wavelet_map: &[f32], offset: usize) -> f32 {
        if wavelet_map.is_empty() {
            return 0.5;
        }
        let index = offset / WAVELET_SAMPLE_INTERVAL;
        wavelet_map.get(index).copied().unwrap_or(0.5)
    }

    // =========================================================================
    // LAZY MODE COMPUTATION
    // Start computation for a mode only when the user switches to it.
    // =========================================================================

    /// Ensure data for the given visualization mode is being computed.
    /// Called when user switches modes - starts computation if not already running.
    fn ensure_mode_data(&mut self, mode: VisualizationMode) {
        let tab = &self.tabs[self.active_tab];
        let Some(ref file) = tab.file_data else {
            return;
        };

        // Check if this mode needs computation and if it's not already done/running
        let needs_computation = match mode {
            VisualizationMode::KolmogorovComplexity => {
                file.complexity_map.is_none()
                    && tab
                        .background_tasks
                        .as_ref()
                        .map(|t| !t.computing_complexity)
                        .unwrap_or(true)
            }
            VisualizationMode::MultiScaleEntropy => {
                file.rcmse_map.is_none()
                    && tab
                        .background_tasks
                        .as_ref()
                        .map(|t| !t.computing_rcmse)
                        .unwrap_or(true)
            }
            VisualizationMode::WaveletEntropy => {
                file.wavelet_map.is_none()
                    && tab
                        .background_tasks
                        .as_ref()
                        .map(|t| !t.computing_wavelet)
                        .unwrap_or(true)
            }
            // Other modes don't need precomputed maps
            _ => false,
        };

        if !needs_computation {
            return;
        }

        // Get or create the background tasks channel
        let data = Arc::clone(&file.data);

        // We need to create a new channel or add to existing one
        // For simplicity, we'll create a new channel and merge receivers
        let (tx, rx) = mpsc::channel();

        match mode {
            VisualizationMode::KolmogorovComplexity => {
                println!("Starting Kolmogorov complexity computation on demand...");
                thread::spawn(move || {
                    Self::precompute_complexity_streaming(&data, &tx);
                });
                let tab = &mut self.tabs[self.active_tab];
                if let Some(ref mut tasks) = tab.background_tasks {
                    tasks.computing_complexity = true;
                } else {
                    tab.background_tasks = Some(BackgroundTasks {
                        receiver: rx,
                        computing_complexity: true,
                        computing_rcmse: false,
                        computing_wavelet: false,
                        computing_wavelet_report: false,
                        complexity_progress: 0.0,
                        rcmse_progress: 0.0,
                        wavelet_progress: 0.0,
                    });
                    return;
                }
            }
            VisualizationMode::MultiScaleEntropy => {
                println!("Starting RCMSE computation on demand...");
                thread::spawn(move || {
                    Self::precompute_rcmse_streaming(&data, &tx);
                });
                let tab = &mut self.tabs[self.active_tab];
                if let Some(ref mut tasks) = tab.background_tasks {
                    tasks.computing_rcmse = true;
                } else {
                    tab.background_tasks = Some(BackgroundTasks {
                        receiver: rx,
                        computing_complexity: false,
                        computing_rcmse: true,
                        computing_wavelet: false,
                        computing_wavelet_report: false,
                        complexity_progress: 0.0,
                        rcmse_progress: 0.0,
                        wavelet_progress: 0.0,
                    });
                    return;
                }
            }
            VisualizationMode::WaveletEntropy => {
                println!("Starting Wavelet entropy computation on demand...");
                thread::spawn(move || {
                    Self::precompute_wavelet_streaming(&data, &tx);
                });
                let tab = &mut self.tabs[self.active_tab];
                if let Some(ref mut tasks) = tab.background_tasks {
                    tasks.computing_wavelet = true;
                } else {
                    tab.background_tasks = Some(BackgroundTasks {
                        receiver: rx,
                        computing_complexity: false,
                        computing_rcmse: false,
                        computing_wavelet: true,
                        computing_wavelet_report: false,
                        complexity_progress: 0.0,
                        rcmse_progress: 0.0,
                        wavelet_progress: 0.0,
                    });
                    return;
                }
            }
            _ => {}
        }

        // Store the new receiver - we need to poll from multiple receivers
        // For now, replace the receiver (existing tasks will complete but won't be polled)
        // TODO: Could merge receivers for better handling
        let tab = &mut self.tabs[self.active_tab];
        if let Some(ref mut tasks) = tab.background_tasks {
            // Replace receiver - old tasks continue but we poll new one
            // This is a simplification; a production version might merge channels
            tasks.receiver = rx;
        }
    }

    /// Pause Hilbert refinement (when switching away from Hilbert mode).
    fn pause_hilbert_refiner(&self) {
        let tab = self.active_tab();
        if let Some(ref file) = tab.file_data {
            if let Some(ref buffer) = file.hilbert_buffer {
                buffer.pause();
            }
        }
    }

    /// Resume Hilbert refinement (when switching back to Hilbert mode).
    fn resume_hilbert_refiner(&self) {
        let tab = self.active_tab();
        if let Some(ref file) = tab.file_data {
            if let Some(ref buffer) = file.hilbert_buffer {
                buffer.resume();
            }
        }
    }

    // =========================================================================
    // STREAMING PRECOMPUTE FUNCTIONS
    // These compute maps in chunks and send partial updates for progressive rendering.
    // =========================================================================

    /// Batch size for streaming updates (number of samples per batch).
    /// Larger = fewer updates but chunkier progress. Smaller = smoother but more overhead.
    const STREAMING_BATCH_SIZE: usize = 100_000;

    /// Precompute complexity map with streaming updates.
    /// Uses SEQUENTIAL iteration to avoid competing with rendering for rayon's thread pool.
    fn precompute_complexity_streaming(data: &[u8], tx: &mpsc::Sender<BackgroundTask>) {
        const WINDOW_SIZE: usize = 128;
        let num_samples = (data.len() / COMPLEXITY_SAMPLE_INTERVAL).max(1);
        let batch_size = Self::STREAMING_BATCH_SIZE;
        let num_batches = (num_samples + batch_size - 1) / batch_size;

        let mut all_results = Vec::with_capacity(num_samples);

        for batch_idx in 0..num_batches {
            let batch_start = batch_idx * batch_size;
            let batch_end = (batch_start + batch_size).min(num_samples);

            // Compute this batch SEQUENTIALLY to avoid rayon contention with rendering
            for i in batch_start..batch_end {
                let start = i * COMPLEXITY_SAMPLE_INTERVAL;
                let end = (start + WINDOW_SIZE).min(data.len());
                let value = if start < data.len() {
                    calculate_kolmogorov_complexity(&data[start..end]) as f32
                } else {
                    0.0
                };
                all_results.push(value);
            }

            // Send progress update
            let progress = batch_end as f32 / num_samples as f32;
            let _ = tx.send(BackgroundTask::ComplexityPartial {
                data: all_results.clone(),
                progress,
            });
        }

        let _ = tx.send(BackgroundTask::ComplexityComplete);
    }

    /// Precompute RCMSE map with streaming updates.
    /// Uses SEQUENTIAL iteration to avoid competing with rendering for rayon's thread pool.
    fn precompute_rcmse_streaming(data: &[u8], tx: &mpsc::Sender<BackgroundTask>) {
        const WINDOW_SIZE: usize = 256;
        let num_samples = (data.len() / RCMSE_SAMPLE_INTERVAL).max(1);
        let batch_size = Self::STREAMING_BATCH_SIZE;
        let num_batches = (num_samples + batch_size - 1) / batch_size;

        let mut all_results = Vec::with_capacity(num_samples);

        for batch_idx in 0..num_batches {
            let batch_start = batch_idx * batch_size;
            let batch_end = (batch_start + batch_size).min(num_samples);

            // Compute this batch SEQUENTIALLY to avoid rayon contention with rendering
            for i in batch_start..batch_end {
                let start = i * RCMSE_SAMPLE_INTERVAL;
                let end = (start + WINDOW_SIZE).min(data.len());
                let value = if start < data.len() && end > start {
                    calculate_rcmse_quick(&data[start..end])
                } else {
                    0.5
                };
                all_results.push(value);
            }

            // Send progress update
            let progress = batch_end as f32 / num_samples as f32;
            let _ = tx.send(BackgroundTask::RcmsePartial {
                data: all_results.clone(),
                progress,
            });
        }

        let _ = tx.send(BackgroundTask::RcmseComplete);
    }

    /// Precompute wavelet map with streaming updates using parallel batches.
    fn precompute_wavelet_streaming(data: &[u8], tx: &mpsc::Sender<BackgroundTask>) {
        const WINDOW_SIZE: usize = 2048;
        const CHUNKS_PER_WINDOW: usize = WINDOW_SIZE / WAVELET_CHUNK_SIZE; // 8
        const CHUNK_INDEX_STEP: usize = WAVELET_CHUNK_SIZE / WAVELET_SAMPLE_INTERVAL; // 4

        let total_samples = (data.len() / WAVELET_SAMPLE_INTERVAL).max(1);

        if data.len() < WINDOW_SIZE {
            // File too small - send complete result immediately
            let results = vec![0.5f32; total_samples];
            let _ = tx.send(BackgroundTask::WaveletPartial {
                data: results,
                progress: 1.0,
            });
            let _ = tx.send(BackgroundTask::WaveletComplete);
            return;
        }

        // Step 1: Precompute all chunk entropies first (this is fast, done in parallel)
        let chunk_entropies =
            precompute_chunk_entropies(data, WAVELET_CHUNK_SIZE, WAVELET_SAMPLE_INTERVAL);

        let num_full_samples = (data.len() - WINDOW_SIZE) / WAVELET_SAMPLE_INTERVAL + 1;
        let batch_size = Self::STREAMING_BATCH_SIZE;
        let num_batches = (num_full_samples + batch_size - 1) / batch_size;

        let mut all_results = Vec::with_capacity(total_samples);

        // Step 2: Compute wavelet suspiciousness SEQUENTIALLY to avoid rayon contention
        for batch_idx in 0..num_batches {
            let batch_start = batch_idx * batch_size;
            let batch_end = (batch_start + batch_size).min(num_full_samples);

            // Compute this batch SEQUENTIALLY to avoid rayon contention with rendering
            for sample_idx in batch_start..batch_end {
                let mut stream = [0.0f64; CHUNKS_PER_WINDOW];
                for c in 0..CHUNKS_PER_WINDOW {
                    let chunk_idx = sample_idx + c * CHUNK_INDEX_STEP;
                    stream[c] = chunk_entropies.get(chunk_idx).copied().unwrap_or(0.0);
                }
                all_results.push(wavelet_suspiciousness_8(&stream) as f32);
            }

            // Fill remaining samples with last value for complete visualization
            let last_value = all_results.last().copied().unwrap_or(0.5);
            let mut full_results = all_results.clone();
            full_results.resize(total_samples, last_value);

            // Send progress update
            let progress = batch_end as f32 / num_full_samples as f32;
            let _ = tx.send(BackgroundTask::WaveletPartial {
                data: full_results,
                progress,
            });
        }

        // Ensure final result has all samples
        let last_value = all_results.last().copied().unwrap_or(0.5);
        all_results.resize(total_samples, last_value);

        let _ = tx.send(BackgroundTask::WaveletComplete);
    }

    /// Update the selection based on a byte offset and optionally sync hex view.
    fn update_selection(&mut self, offset: u64) {
        self.update_selection_with_scroll(offset, true);
    }

    /// Update the selection, optionally scrolling the hex view.
    fn update_selection_with_scroll(&mut self, offset: u64, scroll_hex_view: bool) {
        let tab = self.active_tab_mut();
        Self::update_selection_on_tab_with_scroll(tab, offset, scroll_hex_view);
    }

    /// Static helper to update selection on a specific tab.
    fn update_selection_on_tab(tab: &mut Tab, offset: u64) {
        Self::update_selection_on_tab_with_scroll(tab, offset, true);
    }

    /// Static helper to update selection on a specific tab with scroll control.
    fn update_selection_on_tab_with_scroll(tab: &mut Tab, offset: u64, scroll_hex_view: bool) {
        if let Some(file) = &tab.file_data {
            if offset < file.size {
                let start = offset as usize;
                let end = (start + ANALYSIS_WINDOW).min(file.data.len());
                let chunk = &file.data[start..end];

                // Use larger window for Kolmogorov complexity (more meaningful compression)
                const COMPLEXITY_WINDOW: usize = 256;
                let complexity_end = (start + COMPLEXITY_WINDOW).min(file.data.len());
                let complexity_chunk = &file.data[start..complexity_end];

                tab.selection = Selection {
                    offset,
                    entropy: calculate_entropy(chunk),
                    kolmogorov_complexity: calculate_kolmogorov_complexity(complexity_chunk),
                    ascii_string: extract_ascii(chunk),
                };

                // Scroll hex view to show the selection
                if scroll_hex_view {
                    tab.hex_view.scroll_to(offset, file.size);
                }
            }
        }
    }

    /// Handle cursor interaction on the visualization.
    fn handle_interaction(&mut self, world_pos: Vec2) {
        let tab = self.active_tab();
        let Some(file) = &tab.file_data else {
            return;
        };

        if world_pos.x < 0.0 || world_pos.y < 0.0 {
            return;
        }

        let viz_mode = tab.viz_mode;
        let dimension = file.dimension;
        let file_size = file.size;

        // Find the offset based on interaction
        let offset = match viz_mode {
            VisualizationMode::Hilbert
            | VisualizationMode::KolmogorovComplexity
            | VisualizationMode::JensenShannonDivergence
            | VisualizationMode::MultiScaleEntropy
            | VisualizationMode::WaveletEntropy => {
                let x = world_pos.x as u64;
                let y = world_pos.y as u64;

                if x < dimension && y < dimension {
                    let d = xy2d(dimension, x, y);
                    if d < file_size {
                        Some(d)
                    } else {
                        Some(0)
                    }
                } else {
                    Some(0)
                }
            }
            VisualizationMode::Digraph | VisualizationMode::BytePhaseSpace => {
                // World coordinates map to byte values (0-255)
                let byte_x = (world_pos.x as u64).min(255);
                let byte_y = (world_pos.y as u64).min(255);

                // Find first occurrence of this byte pair in the file
                if file_size >= 2 {
                    file.data
                        .windows(2)
                        .enumerate()
                        .find(|(_, window)| {
                            window[0] as u64 == byte_x && window[1] as u64 == byte_y
                        })
                        .map(|(i, _)| i as u64)
                        .or(Some(0))
                } else {
                    Some(0)
                }
            }
            VisualizationMode::SimilarityMatrix => {
                // Map to file positions based on X/Y
                let pos_x = (world_pos.x as u64 * file_size / dimension).min(file_size - 1);
                Some(pos_x)
            }
        };

        if let Some(off) = offset {
            self.update_selection(off);
        }
    }

    /// Generate the entropy visualization texture for the visible viewport.
    fn generate_texture(&mut self, ctx: &egui::Context, view_rect: Rect) {
        let tab = &self.tabs[self.active_tab];
        let Some(file) = &tab.file_data else {
            return;
        };

        // Get world dimension based on visualization mode
        let dim = tab.viz_mode.world_dimension(file.dimension);

        // Calculate world region to render
        let (world_min, world_max) = if tab.viz_mode.renders_full_world() {
            // Render entire world space for consistent sampling
            (Vec2::ZERO, Vec2::new(dim, dim))
        } else {
            // Viewport-aware: only render visible region
            let view_size = view_rect.size();
            let wmin = tab.viewport.offset;
            let wmax = wmin + Vec2::new(view_size.x, view_size.y) / tab.viewport.zoom;
            // Clamp to valid world bounds
            (
                Vec2::new(wmin.x.max(0.0), wmin.y.max(0.0)),
                Vec2::new(wmax.x.min(dim), wmax.y.min(dim)),
            )
        };
        let world_size = world_max - world_min;

        if world_size.x <= 0.0 || world_size.y <= 0.0 {
            return;
        }

        // Determine texture size based on mode requirements
        let max_tex_size = 4096usize;
        let min_tex_size = tab.viz_mode.min_texture_size();

        let tex_size = if tab.viz_mode.renders_full_world() {
            // Use fixed resolution for full-world rendering
            min_tex_size.min(max_tex_size)
        } else {
            // Aim for ~1:1 pixel mapping, respecting min/max bounds
            let pixels_per_world_unit = tab.viewport.zoom;
            let ideal_tex_width = (world_size.x * pixels_per_world_unit) as usize;
            let ideal_tex_height = (world_size.y * pixels_per_world_unit) as usize;
            ideal_tex_width
                .max(ideal_tex_height)
                .clamp(min_tex_size, max_tex_size)
        };

        // Check if we need to regenerate (viewport changed significantly)
        let new_params = TextureParams {
            world_min,
            world_max,
            tex_size,
            viz_mode: tab.viz_mode,
        };

        if let Some(old_params) = &tab.texture_params {
            // Only regenerate if viewport changed significantly or mode changed
            let min_delta = (new_params.world_min - old_params.world_min).length();
            let max_delta = (new_params.world_max - old_params.world_max).length();
            let size_changed = new_params.tex_size != old_params.tex_size;
            let mode_changed = new_params.viz_mode != old_params.viz_mode;

            // Threshold: regenerate if moved more than 10% of visible area or size/mode changed
            let threshold = world_size.length() * 0.1;
            if min_delta < threshold
                && max_delta < threshold
                && !size_changed
                && !mode_changed
                && tab.texture.is_some()
            {
                return;
            }
        }

        let data = Arc::clone(&file.data);
        // Get precomputed maps, using empty vec as fallback if not yet computed
        let complexity_map = file.complexity_map.clone();
        let reference_distribution = Arc::clone(&file.reference_distribution);
        let rcmse_map = file.rcmse_map.clone();
        let wavelet_map = file.wavelet_map.clone();
        let hilbert_buffer = file.hilbert_buffer.clone();
        let dimension = file.dimension;
        let file_size = file.size;
        let viz_mode = tab.viz_mode;

        // Use empty slice as fallback for maps not yet computed
        let empty_map: Arc<Vec<f32>> = Arc::new(Vec::new());
        let complexity_ref = complexity_map.as_ref().unwrap_or(&empty_map);
        let rcmse_ref = rcmse_map.as_ref().unwrap_or(&empty_map);
        let wavelet_ref = wavelet_map.as_ref().unwrap_or(&empty_map);

        // Get current time for progressive rendering animation
        let time_seconds = ctx.input(|i| i.time);

        let scale_x = world_size.x / tex_size as f32;
        let scale_y = world_size.y / tex_size as f32;

        // Try GPU rendering first, fall back to CPU
        // Note: KolmogorovComplexity, JSD, and RCMSE are CPU-only (use precomputed values)
        // Also use CPU for Hilbert when progressive rendering is active (large files)
        let image = if let Some(ref gpu) = self.gpu_renderer {
            if gpu.is_ready() {
                let gpu_mode = match viz_mode {
                    // Use CPU for Hilbert with progressive buffer (shows refinement progress)
                    VisualizationMode::Hilbert if hilbert_buffer.is_some() => None,
                    VisualizationMode::Hilbert => Some(gpu::GpuVizMode::Hilbert),
                    VisualizationMode::Digraph => Some(gpu::GpuVizMode::Digraph),
                    VisualizationMode::BytePhaseSpace => Some(gpu::GpuVizMode::BytePhaseSpace),
                    VisualizationMode::SimilarityMatrix => Some(gpu::GpuVizMode::SimilarityMatrix),
                    // Kolmogorov, JSD, RCMSE, and Wavelet use precomputed/calculated values - CPU rendering
                    VisualizationMode::KolmogorovComplexity
                    | VisualizationMode::JensenShannonDivergence
                    | VisualizationMode::MultiScaleEntropy
                    | VisualizationMode::WaveletEntropy => None,
                };

                if let Some(mode) = gpu_mode {
                    let rgba_data = gpu.render(
                        mode,
                        tex_size as u32,
                        tex_size as u32,
                        world_min.x,
                        world_min.y,
                        world_max.x,
                        world_max.y,
                    );

                    // Convert RGBA bytes to Color32
                    let pixels: Vec<Color32> = rgba_data
                        .chunks_exact(4)
                        .map(|c| Color32::from_rgba_unmultiplied(c[0], c[1], c[2], c[3]))
                        .collect();

                    ColorImage {
                        size: [tex_size, tex_size],
                        pixels,
                    }
                } else {
                    // Mode not supported on GPU, use CPU
                    Self::generate_cpu_image(
                        &data,
                        complexity_ref,
                        &reference_distribution,
                        rcmse_ref,
                        wavelet_ref,
                        hilbert_buffer.as_deref(),
                        dimension,
                        file_size,
                        tex_size,
                        world_min,
                        scale_x,
                        scale_y,
                        viz_mode,
                        time_seconds,
                    )
                }
            } else {
                // GPU not ready, use CPU fallback
                Self::generate_cpu_image(
                    &data,
                    complexity_ref,
                    &reference_distribution,
                    rcmse_ref,
                    wavelet_ref,
                    hilbert_buffer.as_deref(),
                    dimension,
                    file_size,
                    tex_size,
                    world_min,
                    scale_x,
                    scale_y,
                    viz_mode,
                    time_seconds,
                )
            }
        } else {
            // No GPU, use CPU fallback
            Self::generate_cpu_image(
                &data,
                complexity_ref,
                &reference_distribution,
                rcmse_ref,
                wavelet_ref,
                hilbert_buffer.as_deref(),
                dimension,
                file_size,
                tex_size,
                world_min,
                scale_x,
                scale_y,
                viz_mode,
                time_seconds,
            )
        };

        let tab = &mut self.tabs[self.active_tab];
        tab.texture = Some(ctx.load_texture("entropy_map", image, egui::TextureOptions::NEAREST));
        tab.texture_params = Some(new_params);
    }

    /// Generate visualization image using CPU (fallback when GPU unavailable).
    fn generate_cpu_image(
        data: &[u8],
        complexity_map: &[f32],
        reference_distribution: &[f64; 256],
        rcmse_map: &[f32],
        wavelet_map: &[f32],
        hilbert_buffer: Option<&HilbertBuffer>,
        dimension: u64,
        file_size: u64,
        tex_size: usize,
        world_min: Vec2,
        scale_x: f32,
        scale_y: f32,
        viz_mode: VisualizationMode,
        time_seconds: f64,
    ) -> ColorImage {
        let pixels: Vec<Color32> = match viz_mode {
            VisualizationMode::Hilbert => {
                // Use progressive rendering for large files with hilbert_buffer
                if let Some(buffer) = hilbert_buffer {
                    Self::generate_hilbert_pixels_progressive(
                        buffer,
                        dimension,
                        file_size,
                        tex_size,
                        world_min,
                        scale_x,
                        scale_y,
                        time_seconds,
                    )
                } else {
                    Self::generate_hilbert_pixels(
                        data, dimension, file_size, tex_size, world_min, scale_x, scale_y,
                    )
                }
            }
            VisualizationMode::SimilarityMatrix => Self::generate_similarity_matrix_pixels(
                data, file_size, tex_size, world_min, scale_x, scale_y, dimension,
            ),
            VisualizationMode::Digraph => Self::generate_digraph_pixels(
                data, file_size, tex_size, world_min, scale_x, scale_y,
            ),
            VisualizationMode::BytePhaseSpace => Self::generate_byte_phase_space_pixels(
                data, file_size, tex_size, world_min, scale_x, scale_y,
            ),
            VisualizationMode::KolmogorovComplexity => {
                // Show placeholder while computing to avoid parallel rendering competing with background
                if complexity_map.is_empty() {
                    Self::generate_computing_placeholder(tex_size, time_seconds)
                } else {
                    Self::generate_kolmogorov_pixels(
                        complexity_map,
                        dimension,
                        file_size,
                        tex_size,
                        world_min,
                        scale_x,
                        scale_y,
                    )
                }
            }
            VisualizationMode::JensenShannonDivergence => Self::generate_jsd_pixels(
                data,
                reference_distribution,
                dimension,
                file_size,
                tex_size,
                world_min,
                scale_x,
                scale_y,
            ),
            VisualizationMode::MultiScaleEntropy => {
                // Show placeholder while computing to avoid parallel rendering competing with background
                if rcmse_map.is_empty() {
                    Self::generate_computing_placeholder(tex_size, time_seconds)
                } else {
                    Self::generate_rcmse_pixels(
                        rcmse_map, dimension, file_size, tex_size, world_min, scale_x, scale_y,
                    )
                }
            }
            VisualizationMode::WaveletEntropy => {
                // Show placeholder while computing to avoid parallel rendering competing with background
                if wavelet_map.is_empty() {
                    Self::generate_computing_placeholder(tex_size, time_seconds)
                } else {
                    Self::generate_wavelet_pixels(
                        wavelet_map,
                        dimension,
                        file_size,
                        tex_size,
                        world_min,
                        scale_x,
                        scale_y,
                    )
                }
            }
        };

        ColorImage {
            size: [tex_size, tex_size],
            pixels,
        }
    }

    /// Generate pixels using Hilbert curve mapping with entropy coloring.
    /// Used for small files that can be computed immediately.
    fn generate_hilbert_pixels(
        data: &[u8],
        dimension: u64,
        file_size: u64,
        tex_size: usize,
        world_min: Vec2,
        scale_x: f32,
        scale_y: f32,
    ) -> Vec<Color32> {
        (0..tex_size * tex_size)
            .into_par_iter()
            .map(|idx| {
                let tex_y = idx / tex_size;
                let tex_x = idx % tex_size;

                let world_x = (world_min.x + tex_x as f32 * scale_x) as u64;
                let world_y = (world_min.y + tex_y as f32 * scale_y) as u64;

                if world_x >= dimension || world_y >= dimension {
                    return Color32::from_rgb(13, 13, 13);
                }

                let d = xy2d(dimension, world_x, world_y);

                if d >= file_size {
                    return Color32::from_rgb(13, 13, 13);
                }

                let start = d as usize;
                let end = (start + ANALYSIS_WINDOW).min(data.len());
                let chunk = &data[start..end];

                let analysis = ByteAnalysis::analyze(chunk);
                let [r, g, b] = analysis.to_color();
                Color32::from_rgb(r, g, b)
            })
            .collect()
    }

    /// Generate pixels using progressive Hilbert computation buffer.
    /// Shows computed regions with colors, uncomputed regions with animated pulse.
    fn generate_hilbert_pixels_progressive(
        buffer: &HilbertBuffer,
        dimension: u64,
        file_size: u64,
        tex_size: usize,
        world_min: Vec2,
        scale_x: f32,
        scale_y: f32,
        time_seconds: f64,
    ) -> Vec<Color32> {
        use hilbert_refiner::{
            entropy_to_color, entropy_to_color_preview, placeholder_pulse_color,
        };

        (0..tex_size * tex_size)
            .into_par_iter()
            .map(|idx| {
                let tex_y = idx / tex_size;
                let tex_x = idx % tex_size;

                let world_x = (world_min.x + tex_x as f32 * scale_x) as u64;
                let world_y = (world_min.y + tex_y as f32 * scale_y) as u64;

                if world_x >= dimension || world_y >= dimension {
                    return Color32::from_rgb(13, 13, 13);
                }

                let d = xy2d(dimension, world_x, world_y);

                if d >= file_size {
                    return Color32::from_rgb(13, 13, 13);
                }

                // Look up from progressive buffer
                match buffer.lookup(d) {
                    Some((entropy, is_fine)) => {
                        let (r, g, b) = if is_fine {
                            entropy_to_color(entropy)
                        } else {
                            entropy_to_color_preview(entropy)
                        };
                        Color32::from_rgb(r, g, b)
                    }
                    None => {
                        // Not yet computed - show animated pulse
                        let (r, g, b) = placeholder_pulse_color(time_seconds);
                        Color32::from_rgb(r, g, b)
                    }
                }
            })
            .collect()
    }

    /// Generate a simple animated placeholder while data is being computed.
    /// Uses NO parallel iteration to avoid competing with background computation.
    fn generate_computing_placeholder(tex_size: usize, time_seconds: f64) -> Vec<Color32> {
        let total = tex_size * tex_size;
        let mut pixels = Vec::with_capacity(total);

        // Gentle animated gradient to indicate "working"
        let pulse = ((time_seconds * 2.0 * std::f64::consts::PI).sin() * 0.5 + 0.5) as f32;
        let base_color = 25.0 + pulse * 10.0;

        for idx in 0..total {
            let y = idx / tex_size;
            let x = idx % tex_size;

            // Subtle diagonal gradient pattern
            let gradient = ((x + y) as f32 / (tex_size * 2) as f32) * 15.0;
            let v = (base_color + gradient) as u8;

            // Slight blue tint to indicate "computing"
            pixels.push(Color32::from_rgb(v, v, v.saturating_add(12)));
        }

        pixels
    }

    /// Generate Structural Similarity Matrix (Recurrence Plot) with viewport support.
    ///
    /// Each pixel (i, j) shows the similarity between byte windows at
    /// positions i and j in the file. Supports pan/zoom for detailed exploration.
    ///
    /// - Diagonal lines: Repeating patterns/sequences
    /// - Vertical/horizontal lines: Laminar states (unchanged regions)
    /// - Checkerboard patterns: Periodic structures
    fn generate_similarity_matrix_pixels(
        data: &[u8],
        file_size: u64,
        tex_size: usize,
        world_min: Vec2,
        scale_x: f32,
        scale_y: f32,
        dimension: u64,
    ) -> Vec<Color32> {
        let window_size: usize = 32; // Comparison window
        let dim = dimension as f64;

        // Precompute histograms for file positions we'll need
        // Map world coordinates to file positions
        let get_histogram = |file_pos: usize| -> [u32; 256] {
            let mut hist = [0u32; 256];
            let end = (file_pos + window_size).min(data.len());
            if file_pos < data.len() {
                for &byte in &data[file_pos..end] {
                    hist[byte as usize] += 1;
                }
            }
            hist
        };

        (0..tex_size * tex_size)
            .into_par_iter()
            .map(|idx| {
                let tex_y = idx / tex_size;
                let tex_x = idx % tex_size;

                // Map texture coordinates to world coordinates
                let world_x = world_min.x + tex_x as f32 * scale_x;
                let world_y = world_min.y + tex_y as f32 * scale_y;

                if world_x < 0.0 || world_y < 0.0 || world_x >= dim as f32 || world_y >= dim as f32
                {
                    return Color32::from_rgb(10, 10, 15);
                }

                // Map world coordinates to file positions
                let pos_x = ((world_x as f64 / dim) * file_size as f64) as usize;
                let pos_y = ((world_y as f64 / dim) * file_size as f64) as usize;

                let hist_x = get_histogram(pos_x);
                let hist_y = get_histogram(pos_y);

                // Compute chi-squared distance (more sensitive to differences than Jaccard)
                let mut chi_sq = 0.0f64;
                let mut total_x = 0u32;
                let mut total_y = 0u32;

                for i in 0..256 {
                    total_x += hist_x[i];
                    total_y += hist_y[i];
                }

                if total_x > 0 && total_y > 0 {
                    for i in 0..256 {
                        let px = hist_x[i] as f64 / total_x as f64;
                        let py = hist_y[i] as f64 / total_y as f64;
                        let sum = px + py;
                        if sum > 0.0 {
                            chi_sq += (px - py).powi(2) / sum;
                        }
                    }
                }

                // Convert distance to similarity (chi_sq ranges from 0 to 2)
                // Apply sqrt for better spread, then invert
                let similarity = 1.0 - (chi_sq / 2.0).sqrt();

                let is_diagonal = (pos_x as i64 - pos_y as i64).abs() < window_size as i64;
                similarity_to_color(similarity, is_diagonal)
            })
            .collect()
    }

    /// Generate Byte Digraph with viewport support.
    ///
    /// Shows byte pair transition frequencies. X=source byte, Y=destination byte.
    /// The digraph is mapped to a 256x256 logical space that can be zoomed/panned.
    fn generate_digraph_pixels(
        data: &[u8],
        file_size: u64,
        tex_size: usize,
        world_min: Vec2,
        scale_x: f32,
        scale_y: f32,
    ) -> Vec<Color32> {
        // Build the digraph: count transitions from byte A to byte B
        let mut digraph = vec![0u64; 256 * 256];
        let mut max_count = 1u64;

        if file_size >= 2 {
            for window in data.windows(2) {
                let from = window[0] as usize;
                let to = window[1] as usize;
                let idx = from * 256 + to;
                digraph[idx] += 1;
                max_count = max_count.max(digraph[idx]);
            }
        }

        let sqrt_max = (max_count as f64).sqrt();

        // Generate texture with viewport support
        // World space is 256x256 (byte values), scale to dimension for consistency
        (0..tex_size * tex_size)
            .into_par_iter()
            .map(|idx| {
                let tex_y = idx / tex_size;
                let tex_x = idx % tex_size;

                // Map texture coordinates to world coordinates
                let world_x = world_min.x + tex_x as f32 * scale_x;
                let world_y = world_min.y + tex_y as f32 * scale_y;

                // Scale world coordinates (which are in dimension-space) to 0-255
                // For digraph, we want to show 256x256 in the center of the view
                let dg_x = world_x as i32;
                let dg_y = world_y as i32;

                if dg_x < 0 || dg_x >= 256 || dg_y < 0 || dg_y >= 256 {
                    return Color32::from_rgb(8, 8, 12);
                }

                let count = digraph[dg_y as usize * 256 + dg_x as usize];

                if count == 0 {
                    Color32::from_rgb(8, 8, 12)
                } else {
                    // Use sqrt for better contrast distribution than log
                    let intensity = ((count as f64).sqrt() / sqrt_max).powf(0.6);

                    // Color by byte values with full spectrum
                    let hue = ((dg_x + dg_y) as f64 / 512.0) * 300.0; // More hue range
                    let sat = 0.85 - intensity * 0.2; // High count = slightly less saturated
                    let val = 0.15 + intensity * 0.85; // Wider brightness range

                    hsv_to_rgb(hue, sat, val)
                }
            })
            .collect()
    }

    /// Generate Byte Phase Space with viewport support.
    ///
    /// Plots byte[i] vs byte[i+1] for all sequential bytes, colored by file position.
    /// This creates a phase space trajectory showing the file's "attractor".
    fn generate_byte_phase_space_pixels(
        data: &[u8],
        file_size: u64,
        tex_size: usize,
        world_min: Vec2,
        scale_x: f32,
        scale_y: f32,
    ) -> Vec<Color32> {
        if file_size < 2 {
            return vec![Color32::from_rgb(10, 10, 15); tex_size * tex_size];
        }

        // Build phase space: for each pixel, store the latest file position that hits it
        // and total count for alpha/intensity
        let mut phase_space = vec![(0u64, 0u64); 256 * 256]; // (last_position, count)

        for (i, window) in data.windows(2).enumerate() {
            let x = window[0] as usize;
            let y = window[1] as usize;
            let idx = y * 256 + x;
            phase_space[idx] = (i as u64, phase_space[idx].1 + 1);
        }

        let max_count = phase_space
            .iter()
            .map(|(_, c)| *c)
            .max()
            .unwrap_or(1)
            .max(1);
        let file_size_f = file_size as f64;
        let sqrt_max = (max_count as f64).sqrt();

        // Generate texture with viewport support
        (0..tex_size * tex_size)
            .into_par_iter()
            .map(|idx| {
                let tex_y = idx / tex_size;
                let tex_x = idx % tex_size;

                // Map texture coordinates to world coordinates
                let world_x = world_min.x + tex_x as f32 * scale_x;
                let world_y = world_min.y + tex_y as f32 * scale_y;

                let ps_x = world_x as i32;
                let ps_y = world_y as i32;

                if ps_x < 0 || ps_x >= 256 || ps_y < 0 || ps_y >= 256 {
                    return Color32::from_rgb(10, 10, 15);
                }

                let (last_pos, count) = phase_space[ps_y as usize * 256 + ps_x as usize];

                if count == 0 {
                    Color32::from_rgb(8, 8, 12)
                } else {
                    // Use sqrt for better contrast distribution
                    let intensity = ((count as f64).sqrt() / sqrt_max).powf(0.5);

                    // Color by byte coordinates for spatial structure
                    let coord_hue = ((ps_x + ps_y) as f64 / 512.0) * 360.0;

                    // Blend position-based hue with coordinate-based hue
                    let position_ratio = last_pos as f64 / file_size_f;
                    let pos_hue = position_ratio * 120.0; // 0 (red) -> 120 (green)

                    let hue = (coord_hue * 0.7 + pos_hue * 0.3) % 360.0;
                    let sat = 0.75 + intensity * 0.2;
                    let val = 0.1 + intensity * 0.9;

                    hsv_to_rgb(hue, sat, val)
                }
            })
            .collect()
    }

    /// Generate Kolmogorov Complexity visualization using Hilbert curve mapping.
    ///
    /// Uses precomputed complexity values for fast rendering.
    /// Each pixel shows the compression ratio of a window starting at that file offset.
    /// This reveals algorithmic complexity patterns - regions that are simple/repetitive
    /// vs. complex/random.
    ///
    /// Color legend (viridis-inspired):
    /// - Purple/blue: Low complexity (highly compressible, simple patterns)
    /// - Teal/green: Medium complexity (structured data)
    /// - Yellow/orange: High complexity (compressed/encrypted data)
    /// - Red/pink: Maximum complexity (truly random data)
    fn generate_kolmogorov_pixels(
        complexity_map: &[f32],
        dimension: u64,
        file_size: u64,
        tex_size: usize,
        world_min: Vec2,
        scale_x: f32,
        scale_y: f32,
    ) -> Vec<Color32> {
        (0..tex_size * tex_size)
            .into_par_iter()
            .map(|idx| {
                let tex_y = idx / tex_size;
                let tex_x = idx % tex_size;

                let world_x = (world_min.x + tex_x as f32 * scale_x) as u64;
                let world_y = (world_min.y + tex_y as f32 * scale_y) as u64;

                if world_x >= dimension || world_y >= dimension {
                    return Color32::from_rgb(13, 13, 13);
                }

                let d = xy2d(dimension, world_x, world_y);

                if d >= file_size {
                    return Color32::from_rgb(13, 13, 13);
                }

                // Use precomputed complexity value
                let complexity = Self::lookup_complexity(complexity_map, d as usize);
                let analysis = KolmogorovAnalysis {
                    complexity: complexity as f64,
                };
                let [r, g, b] = analysis.to_color();
                Color32::from_rgb(r, g, b)
            })
            .collect()
    }

    /// Generate Jensen-Shannon Divergence visualization using Hilbert curve mapping.
    ///
    /// Shows how different each region's byte distribution is from the overall file.
    /// Highlights anomalous regions with unusual byte distributions.
    ///
    /// Color legend:
    /// - Dark blue/purple: Very similar to file average (normal regions)
    /// - Cyan/teal: Slightly different distribution
    /// - Green/yellow: Moderately different (interesting regions)
    /// - Orange/red: Very different (anomalous regions - headers, compressed sections, etc.)
    fn generate_jsd_pixels(
        data: &[u8],
        reference_distribution: &[f64; 256],
        dimension: u64,
        file_size: u64,
        tex_size: usize,
        world_min: Vec2,
        scale_x: f32,
        scale_y: f32,
    ) -> Vec<Color32> {
        (0..tex_size * tex_size)
            .into_par_iter()
            .map(|idx| {
                let tex_y = idx / tex_size;
                let tex_x = idx % tex_size;

                let world_x = (world_min.x + tex_x as f32 * scale_x) as u64;
                let world_y = (world_min.y + tex_y as f32 * scale_y) as u64;

                if world_x >= dimension || world_y >= dimension {
                    return Color32::from_rgb(13, 13, 13);
                }

                let d = xy2d(dimension, world_x, world_y);

                if d >= file_size {
                    return Color32::from_rgb(13, 13, 13);
                }

                let start = d as usize;
                let end = (start + ANALYSIS_WINDOW).min(data.len());
                let chunk = &data[start..end];

                // Calculate JSD against reference distribution
                let jsd = calculate_jsd(chunk, reference_distribution);
                let analysis = JSDAnalysis { divergence: jsd };
                let [r, g, b] = analysis.to_color();
                Color32::from_rgb(r, g, b)
            })
            .collect()
    }

    /// Generate Refined Composite Multi-Scale Entropy (RCMSE) visualization.
    ///
    /// Uses precomputed RCMSE values for fast rendering.
    /// RCMSE reveals complexity across multiple time scales, distinguishing:
    /// - Random/encrypted data (entropy decreases with scale)
    /// - Structured/complex data (entropy constant across scales)
    /// - Deterministic chaos (entropy increases then decreases)
    ///
    /// Color legend:
    /// - Purple/blue: Random/encrypted (steep negative slope)
    /// - Cyan/teal: Compressed data (moderate negative slope)
    /// - Green: Structured complexity (flat slope, 1/f-like)
    /// - Yellow/orange: Chaotic/interesting patterns (positive early slope)
    fn generate_rcmse_pixels(
        rcmse_map: &[f32],
        dimension: u64,
        file_size: u64,
        tex_size: usize,
        world_min: Vec2,
        scale_x: f32,
        scale_y: f32,
    ) -> Vec<Color32> {
        (0..tex_size * tex_size)
            .into_par_iter()
            .map(|idx| {
                let tex_y = idx / tex_size;
                let tex_x = idx % tex_size;

                let world_x = (world_min.x + tex_x as f32 * scale_x) as u64;
                let world_y = (world_min.y + tex_y as f32 * scale_y) as u64;

                if world_x >= dimension || world_y >= dimension {
                    return Color32::from_rgb(13, 13, 13);
                }

                let d = xy2d(dimension, world_x, world_y);

                if d >= file_size {
                    return Color32::from_rgb(13, 13, 13);
                }

                // Use precomputed RCMSE value (0.0-1.0 range)
                let rcmse_value = Self::lookup_rcmse(rcmse_map, d as usize);

                // Map RCMSE value to color
                // 0.0-0.3: Random (purple/blue)
                // 0.3-0.6: Structured (green)
                // 0.6-1.0: Chaotic (yellow/orange)
                let t = rcmse_value.clamp(0.0, 1.0);

                let (r, g, b) = if t < 0.25 {
                    // Random: Purple to blue
                    let s = t / 0.25;
                    (0.4 - s * 0.2, 0.1 + s * 0.2, 0.8 + s * 0.1)
                } else if t < 0.45 {
                    // Transition: Blue to cyan/teal
                    let s = (t - 0.25) / 0.2;
                    (0.2 - s * 0.1, 0.3 + s * 0.3, 0.9 - s * 0.2)
                } else if t < 0.65 {
                    // Structured: Cyan/teal to green
                    let s = (t - 0.45) / 0.2;
                    (0.1, 0.6 + s * 0.2, 0.7 - s * 0.4)
                } else if t < 0.85 {
                    // Chaotic transition: Green to yellow
                    let s = (t - 0.65) / 0.2;
                    (0.1 + s * 0.8, 0.8, 0.3 - s * 0.2)
                } else {
                    // Highly chaotic: Yellow to orange
                    let s = (t - 0.85) / 0.15;
                    (0.9 + s * 0.1, 0.8 - s * 0.2, 0.1)
                };

                Color32::from_rgb((r * 255.0) as u8, (g * 255.0) as u8, (b * 255.0) as u8)
            })
            .collect()
    }

    /// Generate Wavelet Entropy visualization using Hilbert curve mapping.
    ///
    /// Based on "Wavelet Decomposition of Software Entropy Reveals Symptoms of Malicious Code"
    /// Uses precomputed wavelet suspiciousness values (SSECS) for fast rendering.
    ///
    /// Key insight: Malware concentrates entropic energy at COARSE levels (large entropy shifts
    /// from encrypted/compressed sections), while clean files concentrate energy at FINE levels.
    ///
    /// Color legend:
    /// - Deep blue/cyan: Low suspiciousness (normal, energy at fine levels)
    /// - Green/teal: Medium-low suspiciousness
    /// - Yellow/orange: Medium-high suspiciousness
    /// - Red/magenta: High suspiciousness (malware-like, energy at coarse levels)
    fn generate_wavelet_pixels(
        wavelet_map: &[f32],
        dimension: u64,
        file_size: u64,
        tex_size: usize,
        world_min: Vec2,
        scale_x: f32,
        scale_y: f32,
    ) -> Vec<Color32> {
        (0..tex_size * tex_size)
            .into_par_iter()
            .map(|idx| {
                let tex_y = idx / tex_size;
                let tex_x = idx % tex_size;

                let world_x = (world_min.x + tex_x as f32 * scale_x) as u64;
                let world_y = (world_min.y + tex_y as f32 * scale_y) as u64;

                if world_x >= dimension || world_y >= dimension {
                    return Color32::from_rgb(13, 13, 13);
                }

                let d = xy2d(dimension, world_x, world_y);

                if d >= file_size {
                    return Color32::from_rgb(13, 13, 13);
                }

                // Use precomputed wavelet suspiciousness value (0.0-1.0 range)
                let suspiciousness = Self::lookup_wavelet(wavelet_map, d as usize);

                // Map suspiciousness to color using WaveletAnalysis color scheme
                let t = suspiciousness.clamp(0.0, 1.0);

                // Cool-to-hot colormap emphasizing suspicious regions
                let (r, g, b) = if t < 0.2 {
                    // Deep blue - very normal (energy at fine levels)
                    let s = t / 0.2;
                    (0.1 + s * 0.1, 0.2 + s * 0.3, 0.7 + s * 0.2)
                } else if t < 0.4 {
                    // Blue to cyan/teal
                    let s = (t - 0.2) / 0.2;
                    (0.2 - s * 0.1, 0.5 + s * 0.3, 0.9 - s * 0.2)
                } else if t < 0.6 {
                    // Cyan to green/yellow
                    let s = (t - 0.4) / 0.2;
                    (0.1 + s * 0.6, 0.8 - s * 0.1, 0.7 - s * 0.5)
                } else if t < 0.8 {
                    // Yellow to orange
                    let s = (t - 0.6) / 0.2;
                    (0.7 + s * 0.3, 0.7 - s * 0.3, 0.2 - s * 0.1)
                } else {
                    // Orange to red/magenta - highly suspicious
                    let s = (t - 0.8) / 0.2;
                    (1.0, 0.4 - s * 0.2, 0.1 + s * 0.3)
                };

                Color32::from_rgb((r * 255.0) as u8, (g * 255.0) as u8, (b * 255.0) as u8)
            })
            .collect()
    }

    /// Reset viewport to default.
    fn reset_viewport(&mut self) {
        let tab = self.active_tab_mut();
        tab.viewport = Viewport::default();
        tab.needs_fit_to_view = true;
        tab.last_fit_view_size = None;
        // Clear texture so it regenerates for the new viewport
        tab.texture = None;
        tab.texture_params = None;
    }

    /// Fit the viewport so the visualization fills the available view.
    fn fit_to_view(&mut self, view_size: Vec2) {
        let tab = &mut self.tabs[self.active_tab];
        if let Some(file) = &tab.file_data {
            let world_dim = tab.viz_mode.world_dimension(file.dimension);

            // Calculate zoom to fit world in view with some padding
            let padding = 0.95; // 95% of view
            let zoom_x = (view_size.x * padding) / world_dim;
            let zoom_y = (view_size.y * padding) / world_dim;
            // No minimum zoom constraint - allow fitting arbitrarily large files
            let zoom = zoom_x.min(zoom_y).max(1e-6); // Just prevent division by zero

            // Center the visualization
            let visible_world = view_size / zoom;
            let offset_x = -(visible_world.x - world_dim) / 2.0;
            let offset_y = -(visible_world.y - world_dim) / 2.0;

            tab.viewport.zoom = zoom;
            tab.viewport.offset = Vec2::new(offset_x, offset_y);

            // Store the view size used for this fit
            tab.last_fit_view_size = Some(view_size);

            // Clear texture to force regeneration with new viewport
            tab.texture = None;
            tab.texture_params = None;
        }
        self.tabs[self.active_tab].needs_fit_to_view = false;
    }

    /// Check if the view size has changed significantly since the last fit.
    fn should_refit(&self, current_view_size: Vec2) -> bool {
        let tab = self.active_tab();
        if let Some(last_size) = tab.last_fit_view_size {
            // Re-fit if view size changed by more than 5%
            let delta_x = (current_view_size.x - last_size.x).abs() / last_size.x.max(1.0);
            let delta_y = (current_view_size.y - last_size.y).abs() / last_size.y.max(1.0);
            delta_x > 0.05 || delta_y > 0.05
        } else {
            false
        }
    }
}

// =============================================================================
// UI Implementation
// =============================================================================

impl eframe::App for ApeironApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        // Load initial file from command-line argument (first frame only)
        if let Some(path) = self.initial_file.take() {
            println!("Loading file from command line: {}", path.display());
            self.load_file(path, false);
        }

        // Poll for completed background tasks
        let task_completed = self.poll_background_tasks();

        // If background tasks are in progress or just completed, request repaint
        let tab_viz_mode = self.active_tab().viz_mode;
        if self.is_computing() || task_completed {
            ctx.request_repaint();
            // Invalidate texture if task completed (new data available)
            if task_completed && self.is_mode_data_ready(tab_viz_mode) {
                self.active_tab_mut().texture = None;
            }
        }

        // Handle progressive Hilbert rendering: request periodic repaints and texture updates
        if self.is_hilbert_refining() {
            // Request repaint at ~10fps for smooth progress animation
            ctx.request_repaint_after(Duration::from_millis(100));
            // Invalidate texture to show refinement progress
            self.active_tab_mut().texture = None;
        }

        // Handle file drops
        let dropped_file = ctx.input(|i| {
            self.is_drop_target = !i.raw.hovered_files.is_empty();
            i.raw.dropped_files.first().and_then(|f| f.path.clone())
        });

        if let Some(path) = dropped_file {
            // If current tab has a file, open in new tab; otherwise load in current tab
            let in_new_tab = self.active_tab().file_data.is_some();
            self.load_file(path, in_new_tab);
        }

        // Note: Texture generation is now done in draw_visualization with viewport info

        // Top toolbar with tabs
        egui::TopBottomPanel::top("toolbar")
            .frame(egui::Frame::none().fill(Color32::from_rgb(28, 28, 32)))
            .show(ctx, |ui| {
                let mut tab_to_close: Option<usize> = None;
                let mut tab_to_activate: Option<usize> = None;

                // Tab bar with drag-and-drop reordering
                let mut tab_rects: Vec<egui::Rect> = Vec::new();
                let mut drop_idx: Option<usize> = None;

                ui.horizontal(|ui| {
                    ui.add_space(6.0);

                    // Get current drag position if dragging
                    let drag_pos = ui.input(|i| i.pointer.hover_pos());
                    let is_dragging_any = self.dragging_tab.is_some();

                    for (i, tab) in self.tabs.iter().enumerate() {
                        let is_active = i == self.active_tab;
                        let is_being_dragged = self.dragging_tab == Some(i);
                        let title = if tab.file_data.is_some() {
                            let t = &tab.title;
                            if t.len() > 24 {
                                format!("{}...", &t[..21])
                            } else {
                                t.clone()
                            }
                        } else {
                            "New Tab".to_string()
                        };

                        let show_close = self.tabs.len() > 1 || tab.file_data.is_some();

                        // Calculate tab size
                        let text_width = ui.fonts(|f| {
                            f.glyph_width(&egui::FontId::proportional(12.0), 'M')
                                * title.len() as f32
                        });
                        let tab_width = text_width + if show_close { 36.0 } else { 24.0 };
                        let tab_height = 28.0;

                        // Allocate space and get response (click and drag)
                        let (rect, response) = ui.allocate_exact_size(
                            egui::vec2(tab_width, tab_height),
                            Sense::click_and_drag(),
                        );
                        tab_rects.push(rect);

                        let is_hovered = response.hovered() && !is_dragging_any;

                        // Start drag
                        if response.drag_started() {
                            self.dragging_tab = Some(i);
                        }

                        // Determine if this is a drop target
                        let is_drop_target =
                            if let (Some(drag_idx), Some(pos)) = (self.dragging_tab, drag_pos) {
                                if drag_idx != i {
                                    let dominated = rect.contains(pos);
                                    if dominated {
                                        drop_idx = Some(i);
                                    }
                                    dominated
                                } else {
                                    false
                                }
                            } else {
                                false
                            };

                        // Background color
                        let bg_color = if is_being_dragged {
                            Color32::from_rgb(55, 55, 65)
                        } else if is_drop_target {
                            Color32::from_rgb(50, 60, 70)
                        } else if is_active {
                            Color32::from_rgb(45, 45, 52)
                        } else if is_hovered {
                            Color32::from_rgb(38, 38, 44)
                        } else {
                            Color32::from_rgb(28, 28, 32)
                        };

                        // Draw tab background
                        let rounding = egui::Rounding {
                            nw: 8.0,
                            ne: 8.0,
                            sw: 0.0,
                            se: 0.0,
                        };
                        ui.painter().rect_filled(rect, rounding, bg_color);

                        // Drop indicator line
                        if is_drop_target {
                            if let Some(drag_idx) = self.dragging_tab {
                                let indicator_x = if drag_idx < i {
                                    rect.max.x - 1.0
                                } else {
                                    rect.min.x + 1.0
                                };
                                ui.painter().rect_filled(
                                    egui::Rect::from_min_size(
                                        egui::pos2(indicator_x - 1.5, rect.min.y + 4.0),
                                        egui::vec2(3.0, rect.height() - 8.0),
                                    ),
                                    2.0,
                                    Color32::from_rgb(100, 160, 220),
                                );
                            }
                        }

                        // Active indicator
                        if is_active && !is_being_dragged {
                            ui.painter().rect_filled(
                                egui::Rect::from_min_size(
                                    egui::pos2(rect.min.x, rect.max.y - 2.0),
                                    egui::vec2(rect.width(), 2.0),
                                ),
                                0.0,
                                Color32::from_rgb(70, 130, 180),
                            );
                        }

                        // Tab title
                        let text_color = if is_being_dragged {
                            Color32::from_rgb(180, 180, 190)
                        } else if is_active {
                            Color32::from_rgb(230, 230, 235)
                        } else if is_hovered {
                            Color32::from_rgb(200, 200, 205)
                        } else {
                            Color32::from_rgb(150, 150, 160)
                        };

                        ui.painter().text(
                            egui::pos2(rect.min.x + 10.0, rect.center().y),
                            egui::Align2::LEFT_CENTER,
                            &title,
                            egui::FontId::proportional(12.0),
                            text_color,
                        );

                        // Close button
                        if show_close && !is_being_dragged {
                            let close_rect = egui::Rect::from_center_size(
                                egui::pos2(rect.max.x - 14.0, rect.center().y),
                                egui::vec2(16.0, 16.0),
                            );
                            let pointer_pos =
                                ui.input(|i| i.pointer.hover_pos().unwrap_or_default());
                            let close_hovered = is_hovered && close_rect.contains(pointer_pos);

                            if close_hovered {
                                ui.painter().circle_filled(
                                    close_rect.center(),
                                    8.0,
                                    Color32::from_rgb(65, 65, 72),
                                );
                            }

                            let close_color = if close_hovered {
                                Color32::from_rgb(220, 220, 225)
                            } else if is_hovered || is_active {
                                Color32::from_rgb(130, 130, 140)
                            } else {
                                Color32::TRANSPARENT
                            };

                            if close_color != Color32::TRANSPARENT {
                                ui.painter().text(
                                    close_rect.center(),
                                    egui::Align2::CENTER_CENTER,
                                    "×",
                                    egui::FontId::proportional(14.0),
                                    close_color,
                                );
                            }

                            if response.clicked() && close_hovered {
                                tab_to_close = Some(i);
                            } else if response.clicked() && !is_dragging_any {
                                tab_to_activate = Some(i);
                            }
                        } else if response.clicked() && !is_dragging_any {
                            tab_to_activate = Some(i);
                        }

                        ui.add_space(2.0);
                    }

                    // New tab button
                    ui.add_space(4.0);
                    let (new_rect, new_response) =
                        ui.allocate_exact_size(egui::vec2(24.0, 24.0), Sense::click());

                    let new_hovered = new_response.hovered();
                    if new_hovered {
                        ui.painter().circle_filled(
                            new_rect.center(),
                            11.0,
                            Color32::from_rgb(45, 45, 52),
                        );
                    }

                    ui.painter().text(
                        new_rect.center(),
                        egui::Align2::CENTER_CENTER,
                        "+",
                        egui::FontId::proportional(18.0),
                        if new_hovered {
                            Color32::from_rgb(220, 220, 225)
                        } else {
                            Color32::from_rgb(120, 120, 130)
                        },
                    );

                    if new_response.clicked() {
                        self.new_tab();
                    }
                });

                // Handle drag release - reorder tabs
                if self.dragging_tab.is_some() && ui.input(|i| i.pointer.any_released()) {
                    if let (Some(from), Some(to)) = (self.dragging_tab, drop_idx) {
                        if from != to {
                            let tab = self.tabs.remove(from);
                            let insert_at = if from < to { to } else { to };
                            self.tabs.insert(insert_at, tab);
                            // Update active tab index
                            if self.active_tab == from {
                                self.active_tab = insert_at;
                            } else if from < self.active_tab && self.active_tab <= to {
                                self.active_tab -= 1;
                            } else if to <= self.active_tab && self.active_tab < from {
                                self.active_tab += 1;
                            }
                        }
                    }
                    self.dragging_tab = None;
                }

                // Handle tab actions
                if let Some(i) = tab_to_activate {
                    self.active_tab = i;
                }
                if let Some(i) = tab_to_close {
                    self.close_tab(i);
                }

                // Toolbar row
                ui.add_space(4.0);
                ui.horizontal(|ui| {
                    ui.add_space(8.0);

                    let has_file = self.active_tab().file_data.is_some();

                    // Minimal toolbar buttons
                    let tool_btn = |ui: &mut egui::Ui, text: &str, enabled: bool| -> bool {
                        let response = ui.add_enabled(
                            enabled,
                            egui::Button::new(RichText::new(text).size(11.0).color(if enabled {
                                Color32::from_rgb(175, 175, 185)
                            } else {
                                Color32::from_rgb(80, 80, 90)
                            }))
                            .fill(Color32::from_rgb(40, 40, 46))
                            .rounding(4.0)
                            .min_size(egui::vec2(0.0, 22.0)),
                        );
                        response.clicked()
                    };

                    if tool_btn(ui, "Reset", has_file) {
                        self.reset_viewport();
                    }
                    if tool_btn(ui, "Open", true) {
                        if let Some(path) = rfd::FileDialog::new().pick_file() {
                            let in_new_tab = self.active_tab().file_data.is_some();
                            self.load_file(path, in_new_tab);
                        }
                    }
                    if tool_btn(ui, "?", true) {
                        self.show_help = !self.show_help;
                    }

                    ui.add_space(8.0);

                    // Mode dropdown
                    ui.label(
                        RichText::new("Mode")
                            .size(11.0)
                            .color(Color32::from_rgb(100, 100, 110)),
                    );
                    let old_mode = self.active_tab().viz_mode;
                    let mut new_mode = old_mode;
                    egui::ComboBox::from_id_salt("viz_mode")
                        .selected_text(old_mode.name())
                        .show_ui(ui, |ui| {
                            for mode in VisualizationMode::all() {
                                ui.selectable_value(&mut new_mode, *mode, mode.name());
                            }
                        });

                    // Force texture regeneration and fit viewport when mode changes
                    // (different modes have different world dimensions)
                    if new_mode != old_mode {
                        // Pause/resume Hilbert refinement based on mode
                        if old_mode == VisualizationMode::Hilbert {
                            self.pause_hilbert_refiner();
                        }
                        if new_mode == VisualizationMode::Hilbert {
                            self.resume_hilbert_refiner();
                        }

                        // Start computation for modes that need it (lazy loading)
                        self.ensure_mode_data(new_mode);

                        let tab = self.active_tab_mut();
                        tab.viz_mode = new_mode;
                        tab.texture = None;
                        tab.texture_params = None;
                        // Reset viewport completely to ensure proper fit
                        tab.viewport = Viewport::default();
                        tab.needs_fit_to_view = true;
                        tab.last_fit_view_size = None;
                    }

                    // Show loading indicator when background tasks are running
                    let has_background_tasks = self.active_tab().background_tasks.is_some();
                    let hilbert_progress = self.get_hilbert_refine_progress();
                    let show_hilbert_progress = hilbert_progress.is_some()
                        && self.active_tab().viz_mode == VisualizationMode::Hilbert;

                    if has_background_tasks || show_hilbert_progress {
                        ui.add_space(12.0);

                        // Show progress for each active task
                        ui.horizontal(|ui| {
                            ui.spinner();
                            ui.vertical(|ui| {
                                // Hilbert refinement progress
                                if let Some(progress) = hilbert_progress {
                                    if show_hilbert_progress {
                                        self.draw_task_progress(ui, "Hilbert", progress);
                                    }
                                }
                                // Standard background tasks
                                if let Some(ref tasks) = self.active_tab().background_tasks {
                                    if tasks.computing_complexity {
                                        self.draw_task_progress(
                                            ui,
                                            "Complexity",
                                            tasks.complexity_progress,
                                        );
                                    }
                                    if tasks.computing_rcmse {
                                        self.draw_task_progress(ui, "RCMSE", tasks.rcmse_progress);
                                    }
                                    if tasks.computing_wavelet {
                                        self.draw_task_progress(
                                            ui,
                                            "Wavelet",
                                            tasks.wavelet_progress,
                                        );
                                    }
                                    if tasks.computing_wavelet_report {
                                        ui.label(
                                            RichText::new("Report...")
                                                .size(9.0)
                                                .color(Color32::from_rgb(150, 150, 180)),
                                        );
                                    }
                                }
                            });
                        });
                    }
                });
                ui.add_space(4.0);
            });

        // Right panel: Data Inspector (responsive width)
        egui::SidePanel::right("inspector")
            .min_width(280.0)
            .default_width(280.0)
            .max_width(700.0)
            .show(ctx, |ui| {
                self.draw_inspector(ui);
            });

        // Central panel: Visualization
        egui::CentralPanel::default().show(ctx, |ui| {
            self.draw_visualization(ui);
        });

        // Help popup
        if self.show_help {
            egui::Window::new("Apeiron Guide")
                .collapsible(false)
                .resizable(false)
                .show(ctx, |ui| {
                    self.draw_help(ui);
                });
        }
    }
}

impl ApeironApp {
    /// Draw the entropy visualization panel.
    fn draw_visualization(&mut self, ui: &mut egui::Ui) {
        let available_rect = ui.available_rect_before_wrap();

        // Background
        ui.painter()
            .rect_filled(available_rect, 0.0, Color32::from_rgb(30, 30, 30));

        if self.active_tab().file_data.is_none() {
            // Empty state or drop target indicator
            if self.is_drop_target {
                self.draw_drop_indicator(ui, available_rect);
            } else {
                self.draw_empty_state(ui, available_rect);
            }
            return;
        }

        // Draw drop indicator overlay if dragging
        if self.is_drop_target {
            self.draw_drop_indicator(ui, available_rect);
            return;
        }

        // Fit viewport to view if needed (after load or mode change)
        // Also re-fit if view size changed significantly (handles layout settling, window resize)
        // Only fit if view has valid non-zero size
        let view_size = available_rect.size();
        let needs_fit = self.active_tab().needs_fit_to_view;
        let has_file = self.active_tab().file_data.is_some();
        let should_fit = needs_fit || (has_file && self.should_refit(view_size));

        if should_fit && view_size.x > 100.0 && view_size.y > 100.0 {
            self.fit_to_view(view_size);
        }

        // Interactive area for pan/zoom
        let response = ui.allocate_rect(available_rect, Sense::click_and_drag());

        // Handle zoom (scroll wheel)
        let scroll_delta = ui.input(|i| i.raw_scroll_delta);
        let mut viewport_changed = false;

        if scroll_delta.y != 0.0 && response.hovered() {
            // Calculate dynamic zoom limits based on file/view size
            let (min_zoom, max_zoom) = {
                let tab = self.active_tab();
                if let Some(file) = &tab.file_data {
                    let world_dim = tab.viz_mode.world_dimension(file.dimension);
                    let view_min = available_rect.width().min(available_rect.height());
                    // Min zoom: fit entire world in view (with margin)
                    let min_z = (view_min * 0.8) / world_dim;
                    // Max zoom: ~4 pixels per world unit (very zoomed in)
                    let max_z = 4.0;
                    (min_z.max(1e-6), max_z)
                } else {
                    (0.01, 100.0)
                }
            };

            let tab = self.active_tab_mut();
            let zoom_factor = 1.1f32.powf(scroll_delta.y / 50.0);
            let old_zoom = tab.viewport.zoom;
            let new_zoom = (old_zoom * zoom_factor).clamp(min_zoom, max_zoom);

            // Zoom towards cursor
            if let Some(cursor_pos) = response.hover_pos() {
                let cursor_rel = cursor_pos - available_rect.min;
                let cursor_world_before =
                    Vec2::new(cursor_rel.x, cursor_rel.y) / old_zoom + tab.viewport.offset;
                let cursor_world_after =
                    Vec2::new(cursor_rel.x, cursor_rel.y) / new_zoom + tab.viewport.offset;
                tab.viewport.offset += cursor_world_before - cursor_world_after;
            }

            tab.viewport.zoom = new_zoom;
            viewport_changed = true;
        }

        // Handle pan (drag)
        if response.dragged() {
            let tab = self.active_tab_mut();
            let delta = response.drag_delta();
            tab.viewport.offset -= Vec2::new(delta.x, delta.y) / tab.viewport.zoom;
            viewport_changed = true;
        }

        // Handle hover for inspection
        if let Some(cursor_pos) = response.hover_pos() {
            let tab = self.active_tab();
            let cursor_rel = cursor_pos - available_rect.min;
            let world_pos =
                Vec2::new(cursor_rel.x, cursor_rel.y) / tab.viewport.zoom + tab.viewport.offset;
            self.handle_interaction(world_pos);
        }

        // Generate/update texture for current viewport
        // Skip if mode requires data that isn't ready yet (avoid blocking)
        let current_mode = self.active_tab().viz_mode;
        let data_ready = self.is_mode_data_ready(current_mode);

        // We regenerate when viewport changes or texture is missing, but only if data is ready
        if data_ready && (self.active_tab().texture.is_none() || viewport_changed) {
            self.generate_texture(ui.ctx(), available_rect);
        }

        // Draw the texture
        let tab = self.active_tab();
        if let (Some(texture), Some(params)) = (&tab.texture, &tab.texture_params) {
            let viewport_offset = tab.viewport.offset;
            let viewport_zoom = tab.viewport.zoom;

            // Calculate screen position for the texture's world region
            let world_to_screen = |world: Vec2| -> Pos2 {
                let screen = (world - viewport_offset) * viewport_zoom;
                Pos2::new(
                    screen.x + available_rect.min.x,
                    screen.y + available_rect.min.y,
                )
            };

            let top_left = world_to_screen(params.world_min);
            let bottom_right = world_to_screen(params.world_max);

            let image_rect = Rect::from_min_max(top_left, bottom_right);

            // Only draw if visible
            if image_rect.intersects(available_rect) {
                ui.painter().image(
                    texture.id(),
                    image_rect,
                    Rect::from_min_max(Pos2::ZERO, Pos2::new(1.0, 1.0)),
                    Color32::WHITE,
                );
            }

            // Draw hex view region outline (for Hilbert-based modes)
            let viz_mode = self.active_tab().viz_mode;
            if matches!(
                viz_mode,
                VisualizationMode::Hilbert
                    | VisualizationMode::KolmogorovComplexity
                    | VisualizationMode::JensenShannonDivergence
                    | VisualizationMode::MultiScaleEntropy
                    | VisualizationMode::WaveletEntropy
            ) {
                self.draw_hex_region_outline(ui, available_rect);
            }
        }

        // Draw HUD overlay
        self.draw_hud(ui, available_rect);

        // Draw loading overlay if current mode's data is not ready
        let current_mode = self.active_tab().viz_mode;
        if !self.is_mode_data_ready(current_mode) {
            self.draw_loading_overlay(ui, available_rect);
        }
    }

    /// Draw an outline around the hex view's visible region on the Hilbert curve.
    /// Uses cached outline data for performance.
    fn draw_hex_region_outline(&mut self, ui: &mut egui::Ui, view_rect: Rect) {
        let tab = &self.tabs[self.active_tab];
        let Some(file) = &tab.file_data else {
            return;
        };

        let (start_offset, end_offset) = tab.hex_view.visible_range();
        let end_offset = end_offset.min(file.size);

        if start_offset >= file.size {
            return;
        }

        let dimension = file.dimension;
        let viewport_offset = tab.viewport.offset;
        let viewport_zoom = tab.viewport.zoom;

        // Update cache if needed
        let tab = &mut self.tabs[self.active_tab];
        if !tab
            .hex_view
            .is_cache_valid(start_offset, end_offset, dimension)
        {
            tab.hex_view
                .update_outline_cache(start_offset, end_offset, dimension);
        }

        // Get cached data
        let Some(cache) = &tab.hex_view.outline_cache else {
            return;
        };

        // Convert world coordinates to screen coordinates
        let world_to_screen = |world_x: f32, world_y: f32| -> Pos2 {
            let screen_x = (world_x - viewport_offset.x) * viewport_zoom + view_rect.min.x;
            let screen_y = (world_y - viewport_offset.y) * viewport_zoom + view_rect.min.y;
            Pos2::new(screen_x, screen_y)
        };

        // Convert cached world coordinates to screen coordinates
        let screen_points: Vec<Pos2> = cache
            .points
            .iter()
            .map(|&(x, y)| world_to_screen(x, y))
            .collect();

        // Draw the outline as a polyline with a glow effect
        if screen_points.len() >= 2 {
            // Draw outer glow
            ui.painter().add(egui::Shape::line(
                screen_points.clone(),
                egui::Stroke::new(4.0, Color32::from_rgba_unmultiplied(255, 200, 0, 80)),
            ));
            // Draw main line
            ui.painter().add(egui::Shape::line(
                screen_points,
                egui::Stroke::new(2.0, Color32::from_rgb(255, 220, 50)),
            ));
        }

        // Draw bounding box from cached data
        let (min_x, min_y, max_x, max_y) = cache.bbox;
        if min_x < f32::MAX {
            let box_min = world_to_screen(min_x, min_y);
            let box_max = world_to_screen(max_x, max_y);
            let box_rect = Rect::from_min_max(box_min, box_max);

            // Draw bounding box
            ui.painter().rect_stroke(
                box_rect,
                0.0,
                egui::Stroke::new(1.5, Color32::from_rgba_unmultiplied(255, 220, 50, 150)),
            );
        }
    }

    /// Draw the empty state prompt.
    fn draw_empty_state(&self, ui: &mut egui::Ui, rect: Rect) {
        let center = rect.center();

        ui.painter().text(
            center - Vec2::new(0.0, 20.0),
            egui::Align2::CENTER_CENTER,
            "Drag & Drop a file to begin analyzing",
            egui::FontId::monospace(16.0),
            Color32::GRAY,
        );
    }

    /// Draw loading overlay when current mode's data is still computing.
    fn draw_loading_overlay(&self, ui: &mut egui::Ui, rect: Rect) {
        let tab = self.active_tab();

        // Semi-transparent dark overlay
        ui.painter()
            .rect_filled(rect, 0.0, Color32::from_rgba_unmultiplied(20, 20, 25, 200));

        // Loading message box
        let box_size = Vec2::new(280.0, 100.0);
        let box_rect = Rect::from_center_size(rect.center(), box_size);

        ui.painter()
            .rect_filled(box_rect, 12.0, Color32::from_rgb(35, 35, 45));
        ui.painter().rect_stroke(
            box_rect,
            12.0,
            egui::Stroke::new(2.0, Color32::from_rgb(80, 120, 200)),
        );

        // Mode name
        let mode_name = tab.viz_mode.name();
        ui.painter().text(
            box_rect.center() - Vec2::new(0.0, 20.0),
            egui::Align2::CENTER_CENTER,
            format!("Computing {}", mode_name),
            egui::FontId::monospace(14.0),
            Color32::WHITE,
        );

        // Animated dots (based on time)
        let dots = match (ui.ctx().input(|i| i.time) * 3.0) as i32 % 4 {
            0 => "",
            1 => ".",
            2 => "..",
            _ => "...",
        };
        ui.painter().text(
            box_rect.center() + Vec2::new(0.0, 5.0),
            egui::Align2::CENTER_CENTER,
            format!("Please wait{}", dots),
            egui::FontId::monospace(12.0),
            Color32::GRAY,
        );

        // Progress hint with percentages
        if let Some(ref tasks) = tab.background_tasks {
            // Get the progress for the current mode's relevant task
            let (task_name, progress) = match tab.viz_mode {
                VisualizationMode::KolmogorovComplexity => {
                    ("Complexity", tasks.complexity_progress)
                }
                VisualizationMode::MultiScaleEntropy => ("RCMSE", tasks.rcmse_progress),
                VisualizationMode::WaveletEntropy => ("Wavelet", tasks.wavelet_progress),
                _ => ("Analysis", 0.0),
            };

            let pct = (progress * 100.0) as u32;
            ui.painter().text(
                box_rect.center() + Vec2::new(0.0, 28.0),
                egui::Align2::CENTER_CENTER,
                format!("{}: {}%", task_name, pct),
                egui::FontId::monospace(12.0),
                Color32::from_rgb(180, 160, 100),
            );

            // Progress bar
            let bar_width = 200.0;
            let bar_height = 6.0;
            let bar_rect = Rect::from_center_size(
                box_rect.center() + Vec2::new(0.0, 50.0),
                Vec2::new(bar_width, bar_height),
            );

            // Background
            ui.painter()
                .rect_filled(bar_rect, 3.0, Color32::from_rgb(50, 50, 60));

            // Progress fill
            let fill_width = bar_width * progress;
            let fill_rect = Rect::from_min_size(bar_rect.min, Vec2::new(fill_width, bar_height));
            ui.painter()
                .rect_filled(fill_rect, 3.0, Color32::from_rgb(80, 120, 200));
        }
    }

    /// Draw a compact progress indicator for a background task.
    fn draw_task_progress(&self, ui: &mut egui::Ui, name: &str, progress: f32) {
        let pct = (progress * 100.0) as u32;
        ui.label(
            RichText::new(format!("{}: {}%", name, pct))
                .size(9.0)
                .color(if progress >= 1.0 {
                    Color32::from_rgb(100, 200, 100)
                } else {
                    Color32::from_rgb(180, 160, 100)
                }),
        );
    }

    /// Draw the drop target indicator.
    fn draw_drop_indicator(&self, ui: &mut egui::Ui, rect: Rect) {
        // Darken background
        ui.painter()
            .rect_filled(rect, 0.0, Color32::from_rgba_unmultiplied(0, 0, 0, 150));

        // Dashed border rectangle
        let inner_rect = Rect::from_center_size(rect.center(), Vec2::new(300.0, 200.0));
        ui.painter().rect_stroke(
            inner_rect,
            16.0,
            egui::Stroke::new(4.0, Color32::from_rgb(100, 150, 255)),
        );

        // Text
        ui.painter().text(
            rect.center(),
            egui::Align2::CENTER_CENTER,
            "DROP BINARY FILE",
            egui::FontId::monospace(24.0),
            Color32::WHITE,
        );
    }

    /// Draw the HUD overlay with zoom, position, and file size.
    fn draw_hud(&self, ui: &mut egui::Ui, rect: Rect) {
        let tab = self.active_tab();
        let Some(file) = &tab.file_data else {
            return;
        };

        let hud_rect = Rect::from_min_size(
            rect.min + Vec2::new(16.0, rect.height() - 50.0),
            Vec2::new(380.0, 34.0),
        );

        // Semi-transparent background
        ui.painter().rect_filled(
            hud_rect,
            8.0,
            Color32::from_rgba_unmultiplied(30, 30, 30, 200),
        );

        let mode_indicator = match tab.viz_mode {
            VisualizationMode::Hilbert => "HIL",
            VisualizationMode::SimilarityMatrix => "SIM",
            VisualizationMode::Digraph => "DIG",
            VisualizationMode::BytePhaseSpace => "PHS",
            VisualizationMode::KolmogorovComplexity => "KOL",
            VisualizationMode::JensenShannonDivergence => "JSD",
            VisualizationMode::MultiScaleEntropy => "MSE",
            VisualizationMode::WaveletEntropy => "WAV",
        };

        let text = format!(
            " [{}]  {:.2}x   {:.0}, {:.0}   {}",
            mode_indicator,
            tab.viewport.zoom,
            tab.viewport.offset.x,
            tab.viewport.offset.y,
            format_bytes(file.size)
        );

        ui.painter().text(
            hud_rect.center(),
            egui::Align2::CENTER_CENTER,
            text,
            egui::FontId::monospace(12.0),
            Color32::LIGHT_GRAY,
        );
    }

    /// Draw the data inspector panel with interactive hex view.
    fn draw_inspector(&mut self, ui: &mut egui::Ui) {
        if self.active_tab().file_data.is_none() {
            // No file loaded placeholder
            ui.centered_and_justified(|ui| {
                ui.label(
                    RichText::new("Drop a file to inspect")
                        .monospace()
                        .color(Color32::DARK_GRAY),
                );
            });
            return;
        }

        // Main content with proper margins
        egui::Frame::none()
            .inner_margin(egui::Margin::symmetric(12.0, 8.0))
            .show(ui, |ui| {
                // Get file data for hex view
                let tab = &self.tabs[self.active_tab];
                let file = tab.file_data.as_ref().unwrap();
                let file_size = file.size;
                let data = Arc::clone(&file.data);
                let file_type = file.file_type;
                let selection_offset = tab.selection.offset;

                // Header: File type and size
                ui.horizontal(|ui| {
                    ui.label(
                        RichText::new(file_type)
                            .monospace()
                            .strong()
                            .color(Color32::LIGHT_BLUE),
                    );
                    ui.label(
                        RichText::new(format!(" - {}", format_bytes(file_size)))
                            .monospace()
                            .color(Color32::GRAY),
                    );
                });
                ui.add_space(4.0);
                ui.separator();
                ui.add_space(4.0);

                // Hex View - dynamically calculate bytes per row based on width
                let panel_width = ui.available_width();
                // Calculate bytes_per_row based on available width
                // Approximate: offset(70) + gap(8) + hex(N*22) + gap(4) + hex(N*22) + gap(8) + ascii(N*8)
                // Simplified: 90 + N*30, so N = (width - 90) / 30
                let bytes_per_row = if panel_width >= 500.0 {
                    16
                } else if panel_width >= 380.0 {
                    12
                } else if panel_width >= 260.0 {
                    8
                } else {
                    4
                };

                let row_height = 18.0;
                let visible_rows = 15;
                let hex_view_height = visible_rows as f32 * row_height;

                // Calculate which rows to display, centered on selection
                let selection_row = (selection_offset as usize) / bytes_per_row;
                let half_visible = visible_rows / 2;
                let total_rows = ((file_size as usize + bytes_per_row - 1) / bytes_per_row).max(1);

                // Calculate start row, keeping selection centered
                let start_row = if selection_row < half_visible {
                    0
                } else if selection_row + half_visible >= total_rows {
                    total_rows.saturating_sub(visible_rows)
                } else {
                    selection_row - half_visible
                };

                let end_row = (start_row + visible_rows).min(total_rows);

                // Update hex view state
                {
                    let tab = &mut self.tabs[self.active_tab];
                    tab.hex_view.bytes_per_row = bytes_per_row;
                    tab.hex_view.visible_rows = visible_rows;
                    tab.hex_view.scroll_offset = (start_row * bytes_per_row) as u64;
                }

                // Hex View with proper clipping via ScrollArea
                egui::Frame::none()
                    .fill(Color32::from_rgb(15, 15, 20))
                    .rounding(4.0)
                    .inner_margin(8.0)
                    .show(ui, |ui| {
                        egui::ScrollArea::vertical()
                            .max_height(hex_view_height)
                            .auto_shrink([false, false])
                            .show(ui, |ui| {
                                let half_bytes = bytes_per_row / 2;

                                // Render only the visible rows
                                for row_idx in start_row..end_row {
                                    let row_offset = row_idx * bytes_per_row;
                                    if row_offset >= file_size as usize {
                                        break;
                                    }

                                    let row_end =
                                        (row_offset + bytes_per_row).min(file_size as usize);
                                    let row_bytes = &data[row_offset..row_end];
                                    let is_selected_row = row_idx == selection_row;

                                    ui.horizontal(|ui| {
                                        // Offset column
                                        let offset_color = if is_selected_row {
                                            Color32::YELLOW
                                        } else {
                                            Color32::from_rgb(100, 100, 180)
                                        };
                                        ui.label(
                                            RichText::new(format!("{:08x}", row_offset))
                                                .monospace()
                                                .size(11.0)
                                                .color(offset_color),
                                        );

                                        ui.add_space(6.0);

                                        // Hex bytes - first half
                                        for (i, &byte) in
                                            row_bytes.iter().take(half_bytes).enumerate()
                                        {
                                            let byte_offset = row_offset + i;
                                            let is_cursor_byte =
                                                byte_offset == selection_offset as usize;

                                            if is_cursor_byte {
                                                egui::Frame::none().fill(Color32::YELLOW).show(
                                                    ui,
                                                    |ui| {
                                                        ui.label(
                                                            RichText::new(format!("{:02x}", byte))
                                                                .monospace()
                                                                .size(11.0)
                                                                .color(Color32::BLACK),
                                                        );
                                                    },
                                                );
                                            } else {
                                                ui.label(
                                                    RichText::new(format!("{:02x}", byte))
                                                        .monospace()
                                                        .size(11.0)
                                                        .color(Self::byte_color(byte)),
                                                );
                                            }
                                        }

                                        ui.add_space(4.0);

                                        // Hex bytes - second half
                                        for (i, &byte) in row_bytes
                                            .iter()
                                            .skip(half_bytes)
                                            .take(half_bytes)
                                            .enumerate()
                                        {
                                            let byte_offset = row_offset + half_bytes + i;
                                            let is_cursor_byte =
                                                byte_offset == selection_offset as usize;

                                            if is_cursor_byte {
                                                egui::Frame::none().fill(Color32::YELLOW).show(
                                                    ui,
                                                    |ui| {
                                                        ui.label(
                                                            RichText::new(format!("{:02x}", byte))
                                                                .monospace()
                                                                .size(11.0)
                                                                .color(Color32::BLACK),
                                                        );
                                                    },
                                                );
                                            } else {
                                                ui.label(
                                                    RichText::new(format!("{:02x}", byte))
                                                        .monospace()
                                                        .size(11.0)
                                                        .color(Self::byte_color(byte)),
                                                );
                                            }
                                        }

                                        // Pad if row is incomplete
                                        if row_bytes.len() < bytes_per_row {
                                            let missing = bytes_per_row - row_bytes.len();
                                            ui.label(
                                                RichText::new("   ".repeat(missing))
                                                    .monospace()
                                                    .size(11.0),
                                            );
                                        }

                                        ui.add_space(8.0);

                                        // ASCII column
                                        for (i, &byte) in row_bytes.iter().enumerate() {
                                            let byte_offset = row_offset + i;
                                            let is_cursor_byte =
                                                byte_offset == selection_offset as usize;
                                            let ch = if (0x20..=0x7e).contains(&byte) {
                                                byte as char
                                            } else {
                                                '.'
                                            };

                                            if is_cursor_byte {
                                                ui.label(
                                                    RichText::new(ch.to_string())
                                                        .monospace()
                                                        .size(11.0)
                                                        .color(Color32::BLACK)
                                                        .background_color(Color32::YELLOW),
                                                );
                                            } else {
                                                ui.label(
                                                    RichText::new(ch.to_string())
                                                        .monospace()
                                                        .size(11.0)
                                                        .color(Color32::from_rgb(80, 200, 80)),
                                                );
                                            }
                                        }
                                    });
                                }
                            });
                    });

                ui.add_space(8.0);
                ui.separator();
                ui.add_space(4.0);

                // Get selection values for metrics display
                let tab = &self.tabs[self.active_tab];
                let selection = &tab.selection;
                let selection_offset = selection.offset;
                let selection_entropy = selection.entropy;
                let selection_complexity = selection.kolmogorov_complexity;
                let selection_ascii = selection.ascii_string.clone();
                let wavelet_report = tab.wavelet_report.clone();

                // Metrics section below hex view
                egui::ScrollArea::vertical()
                    .auto_shrink([false, false])
                    .show(ui, |ui| {
                        // Cursor Location
                        Self::section(ui, "CURSOR LOCATION", |ui| {
                            Self::info_row(
                                ui,
                                "OFFSET (HEX)",
                                &format!("0x{:08X}", selection_offset),
                            );
                            Self::info_row(ui, "OFFSET (DEC)", &selection_offset.to_string());
                        });

                        ui.separator();

                        // Entropy Analysis
                        Self::section(ui, "ENTROPY ANALYSIS", |ui| {
                            let available = ui.available_width();
                            let entropy = selection_entropy;
                            ui.horizontal(|ui| {
                                ui.allocate_ui_with_layout(
                                    egui::Vec2::new(available * 0.5, ui.spacing().interact_size.y),
                                    egui::Layout::left_to_right(egui::Align::Center),
                                    |ui| {
                                        ui.label(
                                            RichText::new("ENTROPY")
                                                .monospace()
                                                .color(Color32::GRAY)
                                                .small(),
                                        );
                                    },
                                );
                                ui.allocate_ui_with_layout(
                                    egui::Vec2::new(available * 0.5, ui.spacing().interact_size.y),
                                    egui::Layout::right_to_left(egui::Align::Center),
                                    |ui| {
                                        let entropy_color = Self::entropy_color(entropy);
                                        ui.label(
                                            RichText::new(format!("{:.4}", entropy))
                                                .monospace()
                                                .strong()
                                                .color(entropy_color),
                                        );
                                    },
                                );
                            });

                            // Entropy bar
                            let bar_height = 6.0;
                            let (bar_rect, _) = ui.allocate_exact_size(
                                egui::Vec2::new(ui.available_width(), bar_height),
                                Sense::hover(),
                            );
                            ui.painter()
                                .rect_filled(bar_rect, 3.0, Color32::from_gray(50));
                            let fill_width = bar_rect.width() * (selection_entropy as f32 / 8.0);
                            let fill_rect = Rect::from_min_size(
                                bar_rect.min,
                                egui::Vec2::new(fill_width, bar_height),
                            );
                            ui.painter().rect_filled(
                                fill_rect,
                                3.0,
                                Self::entropy_color(selection_entropy),
                            );

                            // Interpretation
                            let interpretation = if selection_entropy < 1.0 {
                                "Uniform/empty data"
                            } else if selection_entropy < 3.0 {
                                "Low entropy - text/code"
                            } else if selection_entropy < 5.0 {
                                "Medium entropy - mixed"
                            } else if selection_entropy < 7.0 {
                                "High entropy - binary"
                            } else {
                                "Very high - encrypted/compressed"
                            };
                            ui.label(
                                RichText::new(interpretation)
                                    .monospace()
                                    .small()
                                    .color(Color32::GRAY),
                            );
                        });

                        ui.separator();

                        // Kolmogorov Complexity
                        Self::section(ui, "KOLMOGOROV COMPLEXITY", |ui| {
                            let available = ui.available_width();
                            let complexity = selection_complexity;
                            ui.horizontal(|ui| {
                                ui.allocate_ui_with_layout(
                                    egui::Vec2::new(available * 0.5, ui.spacing().interact_size.y),
                                    egui::Layout::left_to_right(egui::Align::Center),
                                    |ui| {
                                        ui.label(
                                            RichText::new("COMPLEXITY")
                                                .monospace()
                                                .color(Color32::GRAY)
                                                .small(),
                                        );
                                    },
                                );
                                ui.allocate_ui_with_layout(
                                    egui::Vec2::new(available * 0.5, ui.spacing().interact_size.y),
                                    egui::Layout::right_to_left(egui::Align::Center),
                                    |ui| {
                                        let complexity_color = Self::complexity_color(complexity);
                                        ui.label(
                                            RichText::new(format!("{:.2}%", complexity * 100.0))
                                                .monospace()
                                                .strong()
                                                .color(complexity_color),
                                        );
                                    },
                                );
                            });

                            // Complexity bar
                            let bar_height = 6.0;
                            let (bar_rect, _) = ui.allocate_exact_size(
                                egui::Vec2::new(ui.available_width(), bar_height),
                                Sense::hover(),
                            );
                            ui.painter()
                                .rect_filled(bar_rect, 3.0, Color32::from_gray(50));
                            let fill_width = bar_rect.width() * selection_complexity as f32;
                            let fill_rect = Rect::from_min_size(
                                bar_rect.min,
                                egui::Vec2::new(fill_width, bar_height),
                            );
                            ui.painter().rect_filled(
                                fill_rect,
                                3.0,
                                Self::complexity_color(selection_complexity),
                            );

                            // Interpretation
                            let interpretation = if selection_complexity < 0.2 {
                                "Highly compressible"
                            } else if selection_complexity < 0.4 {
                                "Simple patterns"
                            } else if selection_complexity < 0.6 {
                                "Structured data"
                            } else if selection_complexity < 0.8 {
                                "Complex/compressed"
                            } else {
                                "Random/encrypted"
                            };
                            ui.label(
                                RichText::new(interpretation)
                                    .monospace()
                                    .small()
                                    .color(Color32::GRAY),
                            );
                        });

                        ui.separator();

                        // SSECS Wavelet Entropy Analysis
                        Self::section(ui, "SSECS (WAVELET ENTROPY)", |ui| {
                            if let Some(ref report) = wavelet_report {
                                let ssecs = &report.ssecs;
                                let available = ui.available_width();

                                // Malware probability
                                ui.horizontal(|ui| {
                                    ui.allocate_ui_with_layout(
                                        egui::Vec2::new(
                                            available * 0.5,
                                            ui.spacing().interact_size.y,
                                        ),
                                        egui::Layout::left_to_right(egui::Align::Center),
                                        |ui| {
                                            ui.label(
                                                RichText::new("MALWARE PROB")
                                                    .monospace()
                                                    .color(Color32::GRAY)
                                                    .small(),
                                            );
                                        },
                                    );
                                    ui.allocate_ui_with_layout(
                                        egui::Vec2::new(
                                            available * 0.5,
                                            ui.spacing().interact_size.y,
                                        ),
                                        egui::Layout::right_to_left(egui::Align::Center),
                                        |ui| {
                                            let prob_color = match ssecs.classification {
                                                wm::MalwareClassification::Clean => {
                                                    Color32::from_rgb(100, 200, 100)
                                                }
                                                wm::MalwareClassification::Suspicious => {
                                                    Color32::from_rgb(255, 200, 50)
                                                }
                                                wm::MalwareClassification::LikelyMalware => {
                                                    Color32::from_rgb(255, 80, 80)
                                                }
                                                wm::MalwareClassification::Unknown => Color32::GRAY,
                                            };
                                            ui.label(
                                                RichText::new(format!(
                                                    "{:.1}%",
                                                    ssecs.probability_malware * 100.0
                                                ))
                                                .monospace()
                                                .strong()
                                                .color(prob_color),
                                            );
                                        },
                                    );
                                });

                                // Malware probability bar
                                let bar_height = 6.0;
                                let (bar_rect, _) = ui.allocate_exact_size(
                                    egui::Vec2::new(ui.available_width(), bar_height),
                                    Sense::hover(),
                                );
                                ui.painter()
                                    .rect_filled(bar_rect, 3.0, Color32::from_gray(50));
                                let fill_width =
                                    bar_rect.width() * ssecs.probability_malware as f32;
                                let fill_rect = Rect::from_min_size(
                                    bar_rect.min,
                                    egui::Vec2::new(fill_width, bar_height),
                                );
                                ui.painter().rect_filled(
                                    fill_rect,
                                    3.0,
                                    match ssecs.classification {
                                        wm::MalwareClassification::Clean => {
                                            Color32::from_rgb(50, 150, 50)
                                        }
                                        wm::MalwareClassification::Suspicious => {
                                            Color32::from_rgb(200, 150, 50)
                                        }
                                        wm::MalwareClassification::LikelyMalware => {
                                            Color32::from_rgb(200, 50, 50)
                                        }
                                        wm::MalwareClassification::Unknown => Color32::GRAY,
                                    },
                                );

                                // Classification label
                                let classification_text = format!("{:?}", ssecs.classification);
                                let classification_color = match ssecs.classification {
                                    wm::MalwareClassification::Clean => {
                                        Color32::from_rgb(100, 200, 100)
                                    }
                                    wm::MalwareClassification::Suspicious => {
                                        Color32::from_rgb(255, 200, 100)
                                    }
                                    wm::MalwareClassification::LikelyMalware => {
                                        Color32::from_rgb(255, 100, 100)
                                    }
                                    wm::MalwareClassification::Unknown => Color32::GRAY,
                                };
                                ui.label(
                                    RichText::new(classification_text)
                                        .monospace()
                                        .strong()
                                        .color(classification_color),
                                );

                                // Energy ratios
                                ui.add_space(4.0);
                                ui.label(
                                    RichText::new("Energy Distribution")
                                        .monospace()
                                        .small()
                                        .color(Color32::GRAY),
                                );

                                ui.horizontal(|ui| {
                                    ui.label(
                                        RichText::new(format!(
                                            "Coarse: {:.1}%",
                                            ssecs.coarse_energy_ratio * 100.0
                                        ))
                                        .monospace()
                                        .small()
                                        .color(Color32::LIGHT_RED),
                                    );
                                    ui.add_space(8.0);
                                    ui.label(
                                        RichText::new(format!(
                                            "Fine: {:.1}%",
                                            ssecs.fine_energy_ratio * 100.0
                                        ))
                                        .monospace()
                                        .small()
                                        .color(Color32::LIGHT_BLUE),
                                    );
                                });

                                // Wavelet levels
                                ui.add_space(4.0);
                                ui.label(
                                    RichText::new(format!(
                                        "Levels: {}, Chunks: {}",
                                        report.num_wavelet_levels, report.num_entropy_chunks
                                    ))
                                    .monospace()
                                    .small()
                                    .color(Color32::GRAY),
                                );
                            } else {
                                ui.label(
                                    RichText::new("Analyzing...")
                                        .monospace()
                                        .color(Color32::GRAY),
                                );
                            }
                        });

                        // String Preview (if ASCII found)
                        if let Some(ref ascii) = selection_ascii {
                            ui.separator();
                            Self::section(ui, "STRING HINT", |ui| {
                                egui::Frame::none()
                                    .fill(Color32::from_rgba_unmultiplied(0, 0, 0, 100))
                                    .rounding(4.0)
                                    .inner_margin(6.0)
                                    .show(ui, |ui| {
                                        // Truncate long strings for display
                                        let display_str = if ascii.len() > 64 {
                                            format!("{}...", &ascii[..64])
                                        } else {
                                            ascii.clone()
                                        };
                                        ui.label(
                                            RichText::new(display_str)
                                                .monospace()
                                                .size(11.0)
                                                .color(Color32::YELLOW),
                                        );
                                    });
                            });
                        }
                    });
            });
    }

    /// Draw a section with a title.
    fn section(ui: &mut egui::Ui, title: &str, content: impl FnOnce(&mut egui::Ui)) {
        ui.vertical(|ui| {
            ui.label(
                RichText::new(title)
                    .monospace()
                    .small()
                    .strong()
                    .color(Color32::GRAY),
            );
            ui.add_space(8.0);
            content(ui);
        });
        ui.add_space(8.0);
    }

    /// Draw an info row with label and value.
    fn info_row(ui: &mut egui::Ui, label: &str, value: &str) {
        let available = ui.available_width();
        ui.horizontal(|ui| {
            ui.allocate_ui_with_layout(
                egui::Vec2::new(available * 0.5, ui.spacing().interact_size.y),
                egui::Layout::left_to_right(egui::Align::Center),
                |ui| {
                    ui.label(
                        RichText::new(label)
                            .monospace()
                            .small()
                            .color(Color32::GRAY),
                    );
                },
            );
            ui.allocate_ui_with_layout(
                egui::Vec2::new(available * 0.5, ui.spacing().interact_size.y),
                egui::Layout::right_to_left(egui::Align::Center),
                |ui| {
                    ui.label(RichText::new(value).monospace().color(Color32::WHITE));
                },
            );
        });
    }

    /// Get color for a byte value based on its characteristics.
    fn byte_color(byte: u8) -> Color32 {
        if byte == 0 {
            Color32::from_rgb(60, 60, 80) // Null - dark blue-gray
        } else if (0x20..=0x7e).contains(&byte) {
            Color32::from_rgb(180, 180, 220) // Printable ASCII - light
        } else if byte == 0xff {
            Color32::from_rgb(255, 100, 100) // 0xFF - red
        } else if byte > 0x7f {
            Color32::from_rgb(255, 180, 100) // High bytes - orange
        } else {
            Color32::from_rgb(100, 180, 255) // Control chars - blue
        }
    }

    /// Get color for entropy value.
    fn entropy_color(entropy: f64) -> Color32 {
        if entropy > 7.0 {
            Color32::from_rgb(255, 80, 80) // Red
        } else if entropy > 4.0 {
            Color32::from_rgb(80, 255, 80) // Green
        } else {
            Color32::from_rgb(80, 150, 255) // Blue
        }
    }

    /// Get color for Kolmogorov complexity value (0-1).
    fn complexity_color(complexity: f64) -> Color32 {
        let t = complexity.clamp(0.0, 1.0);

        // Viridis-inspired colormap matching the visualization
        let (r, g, b) = if t < 0.2 {
            // Deep purple
            (0.25 + t * 0.25, 0.0 + t * 0.75, 0.5 + t * 1.0)
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

        Color32::from_rgb((r * 255.0) as u8, (g * 255.0) as u8, (b * 255.0) as u8)
    }

    /// Draw the help popup content.
    fn draw_help(&mut self, ui: &mut egui::Ui) {
        ui.set_min_width(320.0);

        ui.vertical(|ui| {
            ui.heading("Controls");
            ui.add_space(8.0);

            Self::control_row(ui, "Scroll", "Zoom in/out");
            Self::control_row(ui, "Drag", "Pan view");
            Self::control_row(ui, "Hover", "Inspect bytes");

            ui.add_space(16.0);
            ui.separator();
            ui.add_space(8.0);

            ui.heading("Visualization Modes");
            ui.add_space(8.0);

            ui.label(
                RichText::new("Hilbert Curve [HIL]")
                    .strong()
                    .monospace()
                    .color(Color32::LIGHT_BLUE),
            );
            ui.label(RichText::new("  Space-filling curve mapping.").small());
            ui.label(
                RichText::new("  Preserves locality - nearby bytes are nearby pixels.").small(),
            );
            ui.label(RichText::new("  Color = entropy/data type.").small());
            ui.add_space(8.0);

            ui.label(
                RichText::new("Similarity Matrix [SIM]")
                    .strong()
                    .monospace()
                    .color(Color32::LIGHT_YELLOW),
            );
            ui.label(RichText::new("  Recurrence plot from nonlinear dynamics.").small());
            ui.label(
                RichText::new("  Pixel (x,y) = similarity between positions x and y.").small(),
            );
            ui.label(RichText::new("  Diagonal lines = repeating patterns.").small());
            ui.add_space(8.0);

            ui.label(
                RichText::new("Byte Digraph [DIG]")
                    .strong()
                    .monospace()
                    .color(Color32::LIGHT_GREEN),
            );
            ui.label(RichText::new("  256x256 byte transition frequencies.").small());
            ui.label(RichText::new("  X = source byte, Y = following byte.").small());
            ui.label(RichText::new("  Bright = frequent transition.").small());
            ui.add_space(8.0);

            ui.label(
                RichText::new("Kolmogorov Complexity [KOL]")
                    .strong()
                    .monospace()
                    .color(Color32::from_rgb(255, 200, 100)),
            );
            ui.label(RichText::new("  Algorithmic complexity via compression.").small());
            ui.label(RichText::new("  Approximates shortest program length.").small());
            ui.label(RichText::new("  Low = simple/repetitive, High = random.").small());
            ui.add_space(8.0);

            ui.label(
                RichText::new("JS Divergence [JSD]")
                    .strong()
                    .monospace()
                    .color(Color32::from_rgb(255, 100, 150)),
            );
            ui.label(RichText::new("  Jensen-Shannon divergence from file average.").small());
            ui.label(RichText::new("  Measures distribution anomalies.").small());
            ui.label(RichText::new("  Low = normal, High = unusual byte dist.").small());
            ui.add_space(8.0);

            ui.label(
                RichText::new("Multi-Scale Entropy [MSE]")
                    .strong()
                    .monospace()
                    .color(Color32::from_rgb(100, 200, 180)),
            );
            ui.label(RichText::new("  RCMSE - complexity across time scales.").small());
            ui.label(RichText::new("  Distinguishes random vs structured data.").small());
            ui.label(RichText::new("  Purple = random, Green = structured.").small());
            ui.add_space(8.0);

            ui.label(
                RichText::new("Wavelet Entropy [WAV]")
                    .strong()
                    .monospace()
                    .color(Color32::from_rgb(255, 150, 100)),
            );
            ui.label(RichText::new("  SSECS - Haar wavelet decomposition.").small());
            ui.label(RichText::new("  Detects malware-like entropy patterns.").small());
            ui.label(RichText::new("  Blue = normal, Red = suspicious.").small());

            ui.add_space(8.0);
            ui.separator();
            ui.add_space(8.0);

            ui.heading("Entropy Color Scale");
            ui.add_space(4.0);

            Self::legend_row(ui, Color32::from_rgb(0, 0, 200), "Low (0-2): Padding/Nulls");
            Self::legend_row(
                ui,
                Color32::from_rgb(0, 200, 200),
                "Med-Low (2-4): ASCII/Text",
            );
            Self::legend_row(
                ui,
                Color32::from_rgb(150, 230, 50),
                "Medium (4-6): Code/Data",
            );
            Self::legend_row(ui, Color32::from_rgb(255, 150, 0), "High (6-7): Compressed");
            Self::legend_row(
                ui,
                Color32::from_rgb(255, 50, 50),
                "Very High (7-8): Encrypted",
            );

            ui.add_space(8.0);
            ui.separator();
            ui.add_space(8.0);

            ui.heading("Kolmogorov Color Scale");
            ui.add_space(4.0);

            Self::legend_row(
                ui,
                Color32::from_rgb(64, 0, 128),
                "Very Low: Highly compressible",
            );
            Self::legend_row(ui, Color32::from_rgb(38, 140, 153), "Low: Simple patterns");
            Self::legend_row(
                ui,
                Color32::from_rgb(191, 204, 51),
                "Medium: Structured data",
            );
            Self::legend_row(
                ui,
                Color32::from_rgb(255, 128, 26),
                "High: Compressed/complex",
            );
            Self::legend_row(
                ui,
                Color32::from_rgb(255, 51, 128),
                "Very High: Random/encrypted",
            );
        });

        ui.add_space(8.0);
        if ui.button("Close").clicked() {
            self.show_help = false;
        }
    }

    /// Draw a control hint row.
    fn control_row(ui: &mut egui::Ui, action: &str, description: &str) {
        ui.horizontal(|ui| {
            ui.label(RichText::new(action).strong().monospace());
            ui.label(RichText::new(description).monospace());
        });
    }

    /// Draw a color legend row.
    fn legend_row(ui: &mut egui::Ui, color: Color32, label: &str) {
        ui.horizontal(|ui| {
            let (rect, _) = ui.allocate_exact_size(Vec2::new(12.0, 12.0), Sense::hover());
            ui.painter().circle_filled(rect.center(), 5.0, color);
            ui.label(RichText::new(label).monospace().small());
        });
    }
}

// =============================================================================
// Color Utilities
// =============================================================================

/// Convert HSV color to RGB Color32.
///
/// - hue: 0-360 degrees
/// - saturation: 0-1
/// - value: 0-1
fn hsv_to_rgb(hue: f64, saturation: f64, value: f64) -> Color32 {
    let h = (hue % 360.0) / 60.0;
    let c = value * saturation;
    let x = c * (1.0 - (h % 2.0 - 1.0).abs());
    let m = value - c;

    let (r, g, b) = if h < 1.0 {
        (c, x, 0.0)
    } else if h < 2.0 {
        (x, c, 0.0)
    } else if h < 3.0 {
        (0.0, c, x)
    } else if h < 4.0 {
        (0.0, x, c)
    } else if h < 5.0 {
        (x, 0.0, c)
    } else {
        (c, 0.0, x)
    };

    Color32::from_rgb(
        ((r + m) * 255.0) as u8,
        ((g + m) * 255.0) as u8,
        ((b + m) * 255.0) as u8,
    )
}

/// Map entropy value (0-8) to a perceptually uniform color.
///
/// Uses a custom colormap optimized for entropy visualization:
/// - 0-2 (low): Blue tones (structured/padding)
/// - 2-5 (medium): Green-yellow (code/data)
/// - 5-7 (high): Orange-red (compressed)
/// - 7-8 (very high): Red-white (encrypted/random)
#[allow(dead_code)]
fn entropy_to_color(entropy: f64) -> Color32 {
    let t = (entropy / 8.0).clamp(0.0, 1.0);

    // Custom colormap: blue -> cyan -> green -> yellow -> orange -> red -> white
    let (r, g, b) = if t < 0.25 {
        // Blue to cyan
        let s = t / 0.25;
        (0.0, s * 0.8, 0.8 + s * 0.2)
    } else if t < 0.5 {
        // Cyan to green-yellow
        let s = (t - 0.25) / 0.25;
        (s * 0.6, 0.8 + s * 0.2, 1.0 - s * 0.8)
    } else if t < 0.75 {
        // Yellow to orange-red
        let s = (t - 0.5) / 0.25;
        (0.6 + s * 0.4, 1.0 - s * 0.5, 0.2 - s * 0.2)
    } else {
        // Red to bright red/white
        let s = (t - 0.75) / 0.25;
        (1.0, 0.5 * s, 0.3 * s)
    };

    Color32::from_rgb((r * 255.0) as u8, (g * 255.0) as u8, (b * 255.0) as u8)
}

/// Map similarity value (0-1) to color for recurrence plots.
/// Uses "inferno" colormap style for better perceptual uniformity.
///
/// - 0 (dissimilar): Black/dark purple
/// - 0.5 (moderate): Red/orange
/// - 1.0 (identical): Bright yellow/white
/// - Diagonal elements get special highlighting
fn similarity_to_color(similarity: f64, is_diagonal: bool) -> Color32 {
    // Apply gamma curve to enhance contrast in the mid-range
    let t = similarity.clamp(0.0, 1.0).powf(0.4); // Gamma < 1 brightens dark areas

    if is_diagonal {
        // Diagonal elements (self-comparison) - bright cyan/white
        return Color32::from_rgb((180.0 + t * 75.0) as u8, (220.0 + t * 35.0) as u8, 255);
    }

    // Inferno-inspired colormap: black -> purple -> red -> orange -> yellow -> white
    let (r, g, b) = if t < 0.2 {
        // Black to dark purple
        let s = t / 0.2;
        (s * 0.25, 0.0, s * 0.35)
    } else if t < 0.4 {
        // Dark purple to red-purple
        let s = (t - 0.2) / 0.2;
        (0.25 + s * 0.45, 0.0, 0.35 + s * 0.1)
    } else if t < 0.6 {
        // Red-purple to orange-red
        let s = (t - 0.4) / 0.2;
        (0.7 + s * 0.25, s * 0.25, 0.45 - s * 0.35)
    } else if t < 0.8 {
        // Orange-red to orange-yellow
        let s = (t - 0.6) / 0.2;
        (0.95 + s * 0.05, 0.25 + s * 0.45, 0.1 + s * 0.1)
    } else {
        // Orange-yellow to bright yellow-white
        let s = (t - 0.8) / 0.2;
        (1.0, 0.7 + s * 0.3, 0.2 + s * 0.7)
    };

    Color32::from_rgb((r * 255.0) as u8, (g * 255.0) as u8, (b * 255.0) as u8)
}

// =============================================================================
// Entry Point
// =============================================================================

fn main() -> eframe::Result<()> {
    // Parse command-line arguments
    let args: Vec<String> = std::env::args().collect();
    let initial_file = if args.len() > 1 {
        let path = PathBuf::from(&args[1]);
        if path.exists() {
            Some(path)
        } else {
            eprintln!("Warning: File not found: {}", args[1]);
            None
        }
    } else {
        None
    };

    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([1200.0, 800.0])
            .with_min_inner_size([800.0, 600.0])
            .with_drag_and_drop(true),
        ..Default::default()
    };

    eframe::run_native(
        "Apeiron",
        options,
        Box::new(move |cc| Ok(Box::new(ApeironApp::new_with_file(cc, initial_file)))),
    )
}
