//! Main application and tab state structures.

use std::path::PathBuf;

use eframe::egui::TextureHandle;

use super::types::{
    BackgroundTasks, FileData, HexView, Selection, TextureParams, Viewport, VisualizationMode,
};
use crate::gpu::GpuRenderer;
use crate::wavelet_malware::WaveletMalwareReport;

// =============================================================================
// Tab
// =============================================================================

/// A single tab containing an open file and its state.
pub struct Tab {
    /// Tab title (file name).
    pub title: String,
    /// Loaded file data.
    pub file_data: Option<FileData>,
    /// Viewport state for pan/zoom.
    pub viewport: Viewport,
    /// Currently selected/hovered byte information.
    pub selection: Selection,
    /// Hex view state.
    pub hex_view: HexView,
    /// Cached wavelet malware analysis for the current file.
    pub wavelet_report: Option<WaveletMalwareReport>,
    /// Cached texture for the entropy visualization.
    pub texture: Option<TextureHandle>,
    /// Texture generation parameters (to detect when regeneration is needed).
    pub texture_params: Option<TextureParams>,
    /// Current visualization mode.
    pub viz_mode: VisualizationMode,
    /// Whether viewport needs to be fitted to view (after load or mode change).
    pub needs_fit_to_view: bool,
    /// View size used for the last fit_to_view call.
    pub last_fit_view_size: Option<eframe::egui::Vec2>,
    /// Background computation tasks.
    pub background_tasks: Option<BackgroundTasks>,
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

// =============================================================================
// ApeironApp
// =============================================================================

/// Main application state.
pub struct ApeironApp {
    /// Open tabs.
    pub tabs: Vec<Tab>,
    /// Index of the currently active tab.
    pub active_tab: usize,
    /// Whether the help popup is shown.
    pub show_help: bool,
    /// Whether a file is being dragged over.
    pub is_drop_target: bool,
    /// GPU renderer for accelerated visualization.
    pub gpu_renderer: Option<GpuRenderer>,
    /// Initial file to load (from command-line argument).
    pub initial_file: Option<PathBuf>,
    /// Tab currently being dragged (index).
    pub dragging_tab: Option<usize>,
}

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
    pub fn new(cc: &eframe::CreationContext<'_>) -> Self {
        Self::new_with_file(cc, None)
    }

    /// Create a new application instance with an optional initial file to load.
    pub fn new_with_file(_cc: &eframe::CreationContext<'_>, initial_file: Option<PathBuf>) -> Self {
        let gpu_renderer = GpuRenderer::new();
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
    pub fn active_tab(&self) -> &Tab {
        &self.tabs[self.active_tab]
    }

    /// Get a mutable reference to the active tab.
    pub fn active_tab_mut(&mut self) -> &mut Tab {
        &mut self.tabs[self.active_tab]
    }

    /// Check if any tab has a file loaded.
    #[allow(dead_code)]
    pub fn has_any_file(&self) -> bool {
        self.tabs.iter().any(|t| t.file_data.is_some())
    }

    /// Open a new tab and make it active.
    pub fn new_tab(&mut self) {
        self.tabs.push(Tab::default());
        self.active_tab = self.tabs.len() - 1;
    }

    /// Close the tab at the given index.
    pub fn close_tab(&mut self, index: usize) {
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
}
