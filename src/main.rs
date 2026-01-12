//! Apeiron - Binary file entropy and complexity visualizer using Hilbert curves.
//!
//! A Rust port of NeuroCore, providing visual analysis of binary files
//! through entropy-based color mapping on a Hilbert curve layout.

#![warn(clippy::all, clippy::pedantic)]
#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]

mod analysis;
mod gpu;
mod hilbert;

use std::path::PathBuf;
use std::sync::Arc;

use eframe::egui::{self, Color32, ColorImage, Pos2, Rect, RichText, Sense, TextureHandle, Vec2};
use rayon::prelude::*;

use analysis::{
    byte_distribution, calculate_entropy, calculate_jsd, calculate_kolmogorov_complexity,
    extract_ascii, format_bytes, identify_file_type, ByteAnalysis, JSDAnalysis, KolmogorovAnalysis,
    ANALYSIS_WINDOW,
};
use hilbert::{calculate_dimension, d2xy, xy2d};

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
            | Self::JensenShannonDivergence => file_dimension as f32,
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
            // Kolmogorov and JSD use Hilbert mapping, same requirements
            Self::KolmogorovComplexity | Self::JensenShannonDivergence => 512,
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
            // Hilbert, Kolmogorov and JSD benefit from viewport-aware rendering for large files
            Self::Hilbert | Self::KolmogorovComplexity | Self::JensenShannonDivergence => false,
        }
    }
}

/// Main application state.
struct NeuroCoreApp {
    /// Loaded file data.
    file_data: Option<FileData>,
    /// Viewport state for pan/zoom.
    viewport: Viewport,
    /// Currently selected/hovered byte information.
    selection: Selection,
    /// Hex view state.
    hex_view: HexView,
    /// Cached texture for the entropy visualization.
    texture: Option<TextureHandle>,
    /// Texture generation parameters (to detect when regeneration is needed).
    texture_params: Option<TextureParams>,
    /// Current visualization mode.
    viz_mode: VisualizationMode,
    /// Whether the help popup is shown.
    show_help: bool,
    /// Whether a file is being dragged over.
    is_drop_target: bool,
    /// Whether viewport needs to be fitted to view (after load or mode change).
    needs_fit_to_view: bool,
    /// GPU renderer for accelerated visualization.
    gpu_renderer: Option<gpu::GpuRenderer>,
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
    /// Raw file bytes.
    data: Arc<Vec<u8>>,
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
    complexity_map: Arc<Vec<f32>>,
    /// Reference byte distribution for JSD calculation (whole file distribution).
    reference_distribution: Arc<[f64; 256]>,
}

/// Interval for sampling Kolmogorov complexity (every N bytes).
/// Smaller = more precise but slower to precompute.
const COMPLEXITY_SAMPLE_INTERVAL: usize = 64;

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

impl Default for NeuroCoreApp {
    fn default() -> Self {
        Self {
            file_data: None,
            viewport: Viewport::default(),
            selection: Selection::default(),
            hex_view: HexView::new(),
            texture: None,
            texture_params: None,
            viz_mode: VisualizationMode::default(),
            show_help: false,
            is_drop_target: false,
            needs_fit_to_view: false,
            gpu_renderer: None,
        }
    }
}

impl NeuroCoreApp {
    /// Create a new application instance.
    fn new(_cc: &eframe::CreationContext<'_>) -> Self {
        let gpu_renderer = gpu::GpuRenderer::new();
        if gpu_renderer.is_some() {
            println!("GPU acceleration enabled");
        } else {
            println!("GPU acceleration unavailable, using CPU fallback");
        }
        Self {
            gpu_renderer,
            ..Self::default()
        }
    }

    /// Load a file from the given path.
    fn load_file(&mut self, path: PathBuf) {
        match std::fs::read(&path) {
            Ok(data) => {
                let size = data.len() as u64;
                let dimension = calculate_dimension(size);
                let file_type = identify_file_type(&data);

                // Upload to GPU if available
                if let Some(ref mut gpu) = self.gpu_renderer {
                    gpu.upload_file(&data, dimension);
                }

                // Precompute Kolmogorov complexity map (sampled)
                let complexity_map = Self::precompute_complexity_map(&data);
                println!(
                    "Precomputed complexity map: {} samples",
                    complexity_map.len()
                );

                // Calculate reference byte distribution for JSD
                let reference_distribution = Arc::new(byte_distribution(&data));

                let data = Arc::new(data);

                self.file_data = Some(FileData {
                    data,
                    size,
                    dimension,
                    file_type,
                    path,
                    complexity_map: Arc::new(complexity_map),
                    reference_distribution,
                });

                // Reset viewport and selection
                self.viewport = Viewport::default();
                self.texture = None;
                self.texture_params = None;
                self.needs_fit_to_view = true;

                // Initialize selection at offset 0
                self.update_selection(0);

                println!(
                    "Loaded file: Size={} bytes, Dimension={}, Type={}",
                    size, dimension, file_type
                );
            }
            Err(e) => {
                eprintln!("Error loading file: {e}");
            }
        }
    }

    /// Precompute Kolmogorov complexity values for the file.
    /// Samples every COMPLEXITY_SAMPLE_INTERVAL bytes for performance.
    fn precompute_complexity_map(data: &[u8]) -> Vec<f32> {
        const WINDOW_SIZE: usize = 128;

        let num_samples = (data.len() / COMPLEXITY_SAMPLE_INTERVAL).max(1);

        (0..num_samples)
            .into_par_iter()
            .map(|i| {
                let start = i * COMPLEXITY_SAMPLE_INTERVAL;
                let end = (start + WINDOW_SIZE).min(data.len());
                if start < data.len() {
                    calculate_kolmogorov_complexity(&data[start..end]) as f32
                } else {
                    0.0
                }
            })
            .collect()
    }

    /// Look up precomputed complexity value for a file offset.
    fn lookup_complexity(complexity_map: &[f32], offset: usize) -> f32 {
        if complexity_map.is_empty() {
            return 0.0;
        }
        let index = offset / COMPLEXITY_SAMPLE_INTERVAL;
        complexity_map.get(index).copied().unwrap_or(0.0)
    }

    /// Update the selection based on a byte offset and optionally sync hex view.
    fn update_selection(&mut self, offset: u64) {
        self.update_selection_with_scroll(offset, true);
    }

    /// Update the selection, optionally scrolling the hex view.
    fn update_selection_with_scroll(&mut self, offset: u64, scroll_hex_view: bool) {
        if let Some(file) = &self.file_data {
            if offset < file.size {
                let start = offset as usize;
                let end = (start + ANALYSIS_WINDOW).min(file.data.len());
                let chunk = &file.data[start..end];

                // Use larger window for Kolmogorov complexity (more meaningful compression)
                const COMPLEXITY_WINDOW: usize = 256;
                let complexity_end = (start + COMPLEXITY_WINDOW).min(file.data.len());
                let complexity_chunk = &file.data[start..complexity_end];

                self.selection = Selection {
                    offset,
                    entropy: calculate_entropy(chunk),
                    kolmogorov_complexity: calculate_kolmogorov_complexity(complexity_chunk),
                    ascii_string: extract_ascii(chunk),
                };

                // Scroll hex view to show the selection
                if scroll_hex_view {
                    self.hex_view.scroll_to(offset, file.size);
                }
            }
        }
    }

    /// Handle cursor interaction on the visualization.
    fn handle_interaction(&mut self, world_pos: Vec2) {
        if let Some(file) = &self.file_data {
            if world_pos.x < 0.0 || world_pos.y < 0.0 {
                return;
            }

            match self.viz_mode {
                VisualizationMode::Hilbert
                | VisualizationMode::KolmogorovComplexity
                | VisualizationMode::JensenShannonDivergence => {
                    let x = world_pos.x as u64;
                    let y = world_pos.y as u64;
                    let n = file.dimension;

                    if x < n && y < n {
                        let d = xy2d(n, x, y);
                        if d < file.size {
                            self.update_selection(d);
                            return;
                        }
                    }
                    self.update_selection(0);
                }
                VisualizationMode::Digraph | VisualizationMode::BytePhaseSpace => {
                    // World coordinates map to byte values (0-255)
                    let byte_x = (world_pos.x as u64).min(255);
                    let byte_y = (world_pos.y as u64).min(255);

                    // Find first occurrence of this byte pair in the file
                    if file.size >= 2 {
                        for (i, window) in file.data.windows(2).enumerate() {
                            if window[0] as u64 == byte_x && window[1] as u64 == byte_y {
                                self.update_selection(i as u64);
                                return;
                            }
                        }
                    }
                    self.update_selection(0);
                }
                VisualizationMode::SimilarityMatrix => {
                    // Map to file positions based on X/Y
                    let n = file.dimension;
                    let pos_x = (world_pos.x as u64 * file.size / n).min(file.size - 1);
                    self.update_selection(pos_x);
                }
            }
        }
    }

    /// Generate the entropy visualization texture for the visible viewport.
    fn generate_texture(&mut self, ctx: &egui::Context, view_rect: Rect) {
        let Some(file) = &self.file_data else {
            return;
        };

        // Get world dimension based on visualization mode
        let dim = self.viz_mode.world_dimension(file.dimension);

        // Calculate world region to render
        let (world_min, world_max) = if self.viz_mode.renders_full_world() {
            // Render entire world space for consistent sampling
            (Vec2::ZERO, Vec2::new(dim, dim))
        } else {
            // Viewport-aware: only render visible region
            let view_size = view_rect.size();
            let wmin = self.viewport.offset;
            let wmax = wmin + Vec2::new(view_size.x, view_size.y) / self.viewport.zoom;
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
        let min_tex_size = self.viz_mode.min_texture_size();

        let tex_size = if self.viz_mode.renders_full_world() {
            // Use fixed resolution for full-world rendering
            min_tex_size.min(max_tex_size)
        } else {
            // Aim for ~1:1 pixel mapping, respecting min/max bounds
            let pixels_per_world_unit = self.viewport.zoom;
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
            viz_mode: self.viz_mode,
        };

        if let Some(old_params) = &self.texture_params {
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
                && self.texture.is_some()
            {
                return;
            }
        }

        let data = Arc::clone(&file.data);
        let complexity_map = Arc::clone(&file.complexity_map);
        let reference_distribution = Arc::clone(&file.reference_distribution);
        let dimension = file.dimension;
        let file_size = file.size;
        let viz_mode = self.viz_mode;

        let scale_x = world_size.x / tex_size as f32;
        let scale_y = world_size.y / tex_size as f32;

        // Try GPU rendering first, fall back to CPU
        // Note: KolmogorovComplexity is CPU-only (uses precomputed values)
        let image = if let Some(ref gpu) = self.gpu_renderer {
            if gpu.is_ready() {
                let gpu_mode = match viz_mode {
                    VisualizationMode::Hilbert => Some(gpu::GpuVizMode::Hilbert),
                    VisualizationMode::Digraph => Some(gpu::GpuVizMode::Digraph),
                    VisualizationMode::BytePhaseSpace => Some(gpu::GpuVizMode::BytePhaseSpace),
                    VisualizationMode::SimilarityMatrix => Some(gpu::GpuVizMode::SimilarityMatrix),
                    // Kolmogorov and JSD use precomputed/calculated values - CPU rendering
                    VisualizationMode::KolmogorovComplexity
                    | VisualizationMode::JensenShannonDivergence => None,
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
                        &complexity_map,
                        &reference_distribution,
                        dimension,
                        file_size,
                        tex_size,
                        world_min,
                        scale_x,
                        scale_y,
                        viz_mode,
                    )
                }
            } else {
                // GPU not ready, use CPU fallback
                Self::generate_cpu_image(
                    &data,
                    &complexity_map,
                    &reference_distribution,
                    dimension,
                    file_size,
                    tex_size,
                    world_min,
                    scale_x,
                    scale_y,
                    viz_mode,
                )
            }
        } else {
            // No GPU, use CPU fallback
            Self::generate_cpu_image(
                &data,
                &complexity_map,
                &reference_distribution,
                dimension,
                file_size,
                tex_size,
                world_min,
                scale_x,
                scale_y,
                viz_mode,
            )
        };

        self.texture = Some(ctx.load_texture("entropy_map", image, egui::TextureOptions::NEAREST));
        self.texture_params = Some(new_params);
    }

    /// Generate visualization image using CPU (fallback when GPU unavailable).
    fn generate_cpu_image(
        data: &[u8],
        complexity_map: &[f32],
        reference_distribution: &[f64; 256],
        dimension: u64,
        file_size: u64,
        tex_size: usize,
        world_min: Vec2,
        scale_x: f32,
        scale_y: f32,
        viz_mode: VisualizationMode,
    ) -> ColorImage {
        let pixels: Vec<Color32> = match viz_mode {
            VisualizationMode::Hilbert => Self::generate_hilbert_pixels(
                data, dimension, file_size, tex_size, world_min, scale_x, scale_y,
            ),
            VisualizationMode::SimilarityMatrix => Self::generate_similarity_matrix_pixels(
                data, file_size, tex_size, world_min, scale_x, scale_y, dimension,
            ),
            VisualizationMode::Digraph => Self::generate_digraph_pixels(
                data, file_size, tex_size, world_min, scale_x, scale_y,
            ),
            VisualizationMode::BytePhaseSpace => Self::generate_byte_phase_space_pixels(
                data, file_size, tex_size, world_min, scale_x, scale_y,
            ),
            VisualizationMode::KolmogorovComplexity => Self::generate_kolmogorov_pixels(
                complexity_map,
                dimension,
                file_size,
                tex_size,
                world_min,
                scale_x,
                scale_y,
            ),
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
        };

        ColorImage {
            size: [tex_size, tex_size],
            pixels,
        }
    }

    /// Generate pixels using Hilbert curve mapping with entropy coloring.
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

    /// Reset viewport to default.
    fn reset_viewport(&mut self) {
        self.viewport = Viewport::default();
        self.needs_fit_to_view = true;
    }

    /// Fit the viewport so the visualization fills the available view.
    fn fit_to_view(&mut self, view_size: Vec2) {
        if let Some(file) = &self.file_data {
            let world_dim = self.viz_mode.world_dimension(file.dimension);

            // Calculate zoom to fit world in view with some padding
            let padding = 0.95; // 95% of view
            let zoom_x = (view_size.x * padding) / world_dim;
            let zoom_y = (view_size.y * padding) / world_dim;
            let zoom = zoom_x.min(zoom_y).max(0.1);

            // Center the visualization
            let visible_world = view_size / zoom;
            let offset_x = -(visible_world.x - world_dim) / 2.0;
            let offset_y = -(visible_world.y - world_dim) / 2.0;

            self.viewport.zoom = zoom;
            self.viewport.offset = Vec2::new(offset_x, offset_y);
        }
        self.needs_fit_to_view = false;
    }
}

// =============================================================================
// UI Implementation
// =============================================================================

impl eframe::App for NeuroCoreApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        // Handle file drops
        ctx.input(|i| {
            self.is_drop_target = !i.raw.hovered_files.is_empty();

            if let Some(file) = i.raw.dropped_files.first() {
                if let Some(path) = &file.path {
                    self.load_file(path.clone());
                }
            }
        });

        // Note: Texture generation is now done in draw_visualization with viewport info

        // Top toolbar
        egui::TopBottomPanel::top("toolbar").show(ctx, |ui| {
            ui.horizontal(|ui| {
                ui.add_space(8.0);

                if ui
                    .add_enabled(self.file_data.is_some(), egui::Button::new("Reset View"))
                    .on_hover_text("Reset zoom and pan")
                    .clicked()
                {
                    self.reset_viewport();
                }

                if ui.button("Help").clicked() {
                    self.show_help = !self.show_help;
                }

                if ui.button("Open File...").clicked() {
                    if let Some(path) = rfd::FileDialog::new().pick_file() {
                        self.load_file(path);
                    }
                }

                ui.add_space(16.0);
                ui.separator();
                ui.add_space(8.0);

                // Visualization mode dropdown
                ui.label(RichText::new("Mode:").monospace().small());
                let old_mode = self.viz_mode;
                egui::ComboBox::from_id_salt("viz_mode")
                    .selected_text(self.viz_mode.name())
                    .show_ui(ui, |ui| {
                        for mode in VisualizationMode::all() {
                            ui.selectable_value(&mut self.viz_mode, *mode, mode.name());
                        }
                    });

                // Force texture regeneration and fit viewport when mode changes
                // (different modes have different world dimensions)
                if self.viz_mode != old_mode {
                    self.texture = None;
                    self.texture_params = None;
                    self.needs_fit_to_view = true;
                }
            });
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
            egui::Window::new("NeuroCore Guide")
                .collapsible(false)
                .resizable(false)
                .show(ctx, |ui| {
                    self.draw_help(ui);
                });
        }
    }
}

impl NeuroCoreApp {
    /// Draw the entropy visualization panel.
    fn draw_visualization(&mut self, ui: &mut egui::Ui) {
        let available_rect = ui.available_rect_before_wrap();

        // Background
        ui.painter()
            .rect_filled(available_rect, 0.0, Color32::from_rgb(30, 30, 30));

        if self.file_data.is_none() {
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
        if self.needs_fit_to_view {
            self.fit_to_view(available_rect.size());
        }

        // Interactive area for pan/zoom
        let response = ui.allocate_rect(available_rect, Sense::click_and_drag());

        // Handle zoom (scroll wheel)
        let scroll_delta = ui.input(|i| i.raw_scroll_delta);
        let mut viewport_changed = false;

        if scroll_delta.y != 0.0 && response.hovered() {
            let zoom_factor = 1.1f32.powf(scroll_delta.y / 50.0);
            let old_zoom = self.viewport.zoom;
            let new_zoom = (old_zoom * zoom_factor).clamp(0.1, 100.0);

            // Zoom towards cursor
            if let Some(cursor_pos) = response.hover_pos() {
                let cursor_rel = cursor_pos - available_rect.min;
                let cursor_world_before =
                    Vec2::new(cursor_rel.x, cursor_rel.y) / old_zoom + self.viewport.offset;
                let cursor_world_after =
                    Vec2::new(cursor_rel.x, cursor_rel.y) / new_zoom + self.viewport.offset;
                self.viewport.offset += cursor_world_before - cursor_world_after;
            }

            self.viewport.zoom = new_zoom;
            viewport_changed = true;
        }

        // Handle pan (drag)
        if response.dragged() {
            let delta = response.drag_delta();
            self.viewport.offset -= Vec2::new(delta.x, delta.y) / self.viewport.zoom;
            viewport_changed = true;
        }

        // Handle hover for inspection
        if let Some(cursor_pos) = response.hover_pos() {
            let cursor_rel = cursor_pos - available_rect.min;
            let world_pos =
                Vec2::new(cursor_rel.x, cursor_rel.y) / self.viewport.zoom + self.viewport.offset;
            self.handle_interaction(world_pos);
        }

        // Generate/update texture for current viewport
        // We regenerate when viewport changes or texture is missing
        if self.texture.is_none() || viewport_changed {
            self.generate_texture(ui.ctx(), available_rect);
        }

        // Draw the texture
        if let (Some(texture), Some(params)) = (&self.texture, &self.texture_params) {
            // Calculate screen position for the texture's world region
            let world_to_screen = |world: Vec2| -> Pos2 {
                let screen = (world - self.viewport.offset) * self.viewport.zoom;
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
            if matches!(
                self.viz_mode,
                VisualizationMode::Hilbert
                    | VisualizationMode::KolmogorovComplexity
                    | VisualizationMode::JensenShannonDivergence
            ) {
                self.draw_hex_region_outline(ui, available_rect);
            }
        }

        // Draw HUD overlay
        self.draw_hud(ui, available_rect);
    }

    /// Draw an outline around the hex view's visible region on the Hilbert curve.
    /// Uses cached outline data for performance.
    fn draw_hex_region_outline(&mut self, ui: &mut egui::Ui, view_rect: Rect) {
        let Some(file) = &self.file_data else {
            return;
        };

        let (start_offset, end_offset) = self.hex_view.visible_range();
        let end_offset = end_offset.min(file.size);

        if start_offset >= file.size {
            return;
        }

        let dimension = file.dimension;

        // Update cache if needed
        if !self
            .hex_view
            .is_cache_valid(start_offset, end_offset, dimension)
        {
            self.hex_view
                .update_outline_cache(start_offset, end_offset, dimension);
        }

        // Get cached data
        let Some(cache) = &self.hex_view.outline_cache else {
            return;
        };

        // Convert world coordinates to screen coordinates
        let world_to_screen = |world_x: f32, world_y: f32| -> Pos2 {
            let screen_x =
                (world_x - self.viewport.offset.x) * self.viewport.zoom + view_rect.min.x;
            let screen_y =
                (world_y - self.viewport.offset.y) * self.viewport.zoom + view_rect.min.y;
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
        let Some(file) = &self.file_data else {
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

        let mode_indicator = match self.viz_mode {
            VisualizationMode::Hilbert => "HIL",
            VisualizationMode::SimilarityMatrix => "SIM",
            VisualizationMode::Digraph => "DIG",
            VisualizationMode::BytePhaseSpace => "PHS",
            VisualizationMode::KolmogorovComplexity => "KOL",
            VisualizationMode::JensenShannonDivergence => "JSD",
        };

        let text = format!(
            " [{}]  {:.2}x   {:.0}, {:.0}   {}",
            mode_indicator,
            self.viewport.zoom,
            self.viewport.offset.x,
            self.viewport.offset.y,
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
        if self.file_data.is_none() {
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
                let file = self.file_data.as_ref().unwrap();
                let file_size = file.size;
                let data = Arc::clone(&file.data);
                let file_type = file.file_type;

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
                self.hex_view.bytes_per_row = bytes_per_row;

                let row_height = 18.0;
                let visible_rows = 15;
                let hex_view_height = visible_rows as f32 * row_height;
                self.hex_view.visible_rows = visible_rows;

                // Calculate which rows to display, centered on selection
                let selection_row = (self.selection.offset as usize) / bytes_per_row;
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

                // Update scroll offset for outline calculation
                self.hex_view.scroll_offset = (start_row * bytes_per_row) as u64;

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
                                                byte_offset == self.selection.offset as usize;

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
                                                byte_offset == self.selection.offset as usize;

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
                                                byte_offset == self.selection.offset as usize;
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

                // Metrics section below hex view
                egui::ScrollArea::vertical()
                    .auto_shrink([false, false])
                    .show(ui, |ui| {
                        // Cursor Location
                        Self::section(ui, "CURSOR LOCATION", |ui| {
                            Self::info_row(
                                ui,
                                "OFFSET (HEX)",
                                &format!("0x{:08X}", self.selection.offset),
                            );
                            Self::info_row(ui, "OFFSET (DEC)", &self.selection.offset.to_string());
                        });

                        ui.separator();

                        // Entropy Analysis
                        Self::section(ui, "ENTROPY ANALYSIS", |ui| {
                            let available = ui.available_width();
                            let entropy = self.selection.entropy;
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
                            let fill_width =
                                bar_rect.width() * (self.selection.entropy as f32 / 8.0);
                            let fill_rect = Rect::from_min_size(
                                bar_rect.min,
                                egui::Vec2::new(fill_width, bar_height),
                            );
                            ui.painter().rect_filled(
                                fill_rect,
                                3.0,
                                Self::entropy_color(self.selection.entropy),
                            );

                            // Interpretation
                            let interpretation = if self.selection.entropy < 1.0 {
                                "Uniform/empty data"
                            } else if self.selection.entropy < 3.0 {
                                "Low entropy - text/code"
                            } else if self.selection.entropy < 5.0 {
                                "Medium entropy - mixed"
                            } else if self.selection.entropy < 7.0 {
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
                            let complexity = self.selection.kolmogorov_complexity;
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
                            let fill_width =
                                bar_rect.width() * self.selection.kolmogorov_complexity as f32;
                            let fill_rect = Rect::from_min_size(
                                bar_rect.min,
                                egui::Vec2::new(fill_width, bar_height),
                            );
                            ui.painter().rect_filled(
                                fill_rect,
                                3.0,
                                Self::complexity_color(self.selection.kolmogorov_complexity),
                            );

                            // Interpretation
                            let interpretation = if self.selection.kolmogorov_complexity < 0.2 {
                                "Highly compressible"
                            } else if self.selection.kolmogorov_complexity < 0.4 {
                                "Simple patterns"
                            } else if self.selection.kolmogorov_complexity < 0.6 {
                                "Structured data"
                            } else if self.selection.kolmogorov_complexity < 0.8 {
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

                        // String Preview (if ASCII found)
                        if let Some(ref ascii) = self.selection.ascii_string {
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
        Box::new(|cc| Ok(Box::new(NeuroCoreApp::new(cc)))),
    )
}
