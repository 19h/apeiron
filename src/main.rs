//! APEIRON - BINARY ANALYSIS SYSTEM
//!
//! GPU-accelerated binary entropy visualizer using Hilbert curves.
//! MIL-SPEC TECHNO-BRUTALISM INTERFACE // REV.03
//!
//! Classification: APEIRON-SYS // UNRESTRICTED

#![warn(clippy::all, clippy::pedantic)]
#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]

mod analysis;
mod app;
mod gpu;
mod hilbert;
mod util;
mod viz;
mod wavelet_malware;

use std::fs::File;
use std::path::PathBuf;
use std::sync::mpsc::{self, TryRecvError};
use std::sync::Arc;
use std::thread;
use std::time::Duration;

use memmap2::Mmap;

use eframe::egui::{self, Color32, ColorImage, Pos2, Rect, RichText, Sense, Vec2};

use analysis::{
    byte_distribution_sampled, calculate_entropy, calculate_jsd, calculate_kolmogorov_complexity,
    calculate_rcmse_quick, extract_ascii, identify_file_type, precompute_chunk_entropies,
    wavelet_suspiciousness_8, ANALYSIS_WINDOW, WAVELET_CHUNK_SIZE,
};
use app::{
    ApeironApp, BackgroundTask, BackgroundTasks, FileData, HexView, Selection, Tab, TextureParams,
    Viewport, VisualizationMode, COMPLEXITY_SAMPLE_INTERVAL, RCMSE_SAMPLE_INTERVAL,
    WAVELET_SAMPLE_INTERVAL,
};
use hilbert::{calculate_dimension, xy2d, HilbertBuffer, HilbertRefiner, PROGRESSIVE_THRESHOLD};
use util::color::{
    ALERT_RED, CAUTION_AMBER, DATA_WHITE, DIM_CYAN, INTERFACE_GRAY, MUTED_TEXT, OPERATIONAL_GREEN,
    PANEL_DARK, TACTICAL_CYAN, VOID_BLACK,
};
use util::format_bytes;
use wavelet_malware as wm;

// =============================================================================
// Application Implementation
// =============================================================================

impl ApeironApp {
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
            let report = wm::analyze_file_for_wavelet(&data_for_report);
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
                                "Wavelet analysis ready: {} levels, {} chunks",
                                report.num_wavelet_levels, report.num_entropy_chunks
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
                    viz::generate_hilbert_pixels_progressive(
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
                    viz::generate_hilbert_pixels(
                        data, dimension, file_size, tex_size, world_min, scale_x, scale_y,
                    )
                }
            }
            VisualizationMode::SimilarityMatrix => viz::generate_similarity_matrix_pixels(
                data, file_size, tex_size, world_min, scale_x, scale_y, dimension,
            ),
            VisualizationMode::Digraph => {
                viz::generate_digraph_pixels(data, file_size, tex_size, world_min, scale_x, scale_y)
            }
            VisualizationMode::BytePhaseSpace => viz::generate_byte_phase_space_pixels(
                data, file_size, tex_size, world_min, scale_x, scale_y,
            ),
            VisualizationMode::KolmogorovComplexity => {
                // Show placeholder while computing to avoid parallel rendering competing with background
                if complexity_map.is_empty() {
                    viz::generate_computing_placeholder(tex_size, time_seconds)
                } else {
                    viz::generate_kolmogorov_pixels(
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
            VisualizationMode::JensenShannonDivergence => viz::generate_jsd_pixels(
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
                    viz::generate_computing_placeholder(tex_size, time_seconds)
                } else {
                    viz::generate_rcmse_pixels(
                        rcmse_map, dimension, file_size, tex_size, world_min, scale_x, scale_y,
                    )
                }
            }
            VisualizationMode::WaveletEntropy => {
                // Show placeholder while computing to avoid parallel rendering competing with background
                if wavelet_map.is_empty() {
                    viz::generate_computing_placeholder(tex_size, time_seconds)
                } else {
                    viz::generate_wavelet_pixels(
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

        // Top toolbar with tabs - MIL-SPEC PANEL
        egui::TopBottomPanel::top("toolbar")
            .frame(egui::Frame::none().fill(PANEL_DARK))
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

                        // MIL-SPEC tab background colors
                        let bg_color = if is_being_dragged {
                            INTERFACE_GRAY
                        } else if is_drop_target {
                            DIM_CYAN.gamma_multiply(0.3) // Tactical cyan tint
                        } else if is_active {
                            INTERFACE_GRAY
                        } else if is_hovered {
                            INTERFACE_GRAY.gamma_multiply(0.6)
                        } else {
                            PANEL_DARK
                        };

                        // Draw tab background - ZERO rounding (MIL-SPEC sharp corners)
                        ui.painter().rect_filled(rect, 0.0, bg_color);

                        // Drop indicator line - TACTICAL CYAN
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
                                    0.0,
                                    TACTICAL_CYAN,
                                );
                            }
                        }

                        // Active indicator - TACTICAL CYAN underline
                        if is_active && !is_being_dragged {
                            ui.painter().rect_filled(
                                egui::Rect::from_min_size(
                                    egui::pos2(rect.min.x, rect.max.y - 2.0),
                                    egui::vec2(rect.width(), 2.0),
                                ),
                                0.0,
                                TACTICAL_CYAN,
                            );
                        }

                        // Tab title - MIL-SPEC text colors
                        let text_color = if is_being_dragged {
                            MUTED_TEXT
                        } else if is_active {
                            DATA_WHITE
                        } else if is_hovered {
                            DATA_WHITE.gamma_multiply(0.8)
                        } else {
                            MUTED_TEXT
                        };

                        ui.painter().text(
                            egui::pos2(rect.min.x + 10.0, rect.center().y),
                            egui::Align2::LEFT_CENTER,
                            &title,
                            egui::FontId::proportional(12.0),
                            text_color,
                        );

                        // Close button - MIL-SPEC
                        if show_close && !is_being_dragged {
                            let close_rect = egui::Rect::from_center_size(
                                egui::pos2(rect.max.x - 14.0, rect.center().y),
                                egui::vec2(16.0, 16.0),
                            );
                            let pointer_pos =
                                ui.input(|i| i.pointer.hover_pos().unwrap_or_default());
                            let close_hovered = is_hovered && close_rect.contains(pointer_pos);

                            if close_hovered {
                                ui.painter().rect_filled(
                                    close_rect,
                                    2.0,
                                    ALERT_RED.gamma_multiply(0.3),
                                );
                            }

                            let close_color = if close_hovered {
                                ALERT_RED
                            } else if is_hovered || is_active {
                                MUTED_TEXT
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

                    // New tab button - MIL-SPEC
                    ui.add_space(4.0);
                    let (new_rect, new_response) =
                        ui.allocate_exact_size(egui::vec2(24.0, 24.0), Sense::click());

                    let new_hovered = new_response.hovered();
                    if new_hovered {
                        ui.painter().rect_filled(new_rect, 2.0, INTERFACE_GRAY);
                    }

                    ui.painter().text(
                        new_rect.center(),
                        egui::Align2::CENTER_CENTER,
                        "+",
                        egui::FontId::proportional(18.0),
                        if new_hovered {
                            TACTICAL_CYAN
                        } else {
                            MUTED_TEXT
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

                // Toolbar row - MIL-SPEC CONTROLS
                ui.add_space(4.0);
                ui.horizontal(|ui| {
                    ui.add_space(8.0);

                    let has_file = self.active_tab().file_data.is_some();

                    // MIL-SPEC toolbar buttons - ZERO rounding
                    let tool_btn = |ui: &mut egui::Ui, text: &str, enabled: bool| -> bool {
                        let response = ui.add_enabled(
                            enabled,
                            egui::Button::new(RichText::new(text.to_uppercase()).size(10.0).color(
                                if enabled {
                                    DATA_WHITE
                                } else {
                                    MUTED_TEXT.gamma_multiply(0.5) // Disabled state
                                },
                            ))
                            .fill(INTERFACE_GRAY)
                            .rounding(0.0)
                            .min_size(egui::vec2(0.0, 22.0)),
                        );
                        response.clicked()
                    };

                    if tool_btn(ui, "RESET", has_file) {
                        self.reset_viewport();
                    }
                    if tool_btn(ui, "OPEN", true) {
                        if let Some(path) = rfd::FileDialog::new().pick_file() {
                            let in_new_tab = self.active_tab().file_data.is_some();
                            self.load_file(path, in_new_tab);
                        }
                    }
                    if tool_btn(ui, "?", true) {
                        self.show_help = !self.show_help;
                    }

                    ui.add_space(8.0);

                    // Mode dropdown - MIL-SPEC label
                    ui.label(RichText::new("MODE:").size(10.0).color(MUTED_TEXT));
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
                                            RichText::new("[ANALYZING]")
                                                .size(9.0)
                                                .color(TACTICAL_CYAN),
                                        );
                                    }
                                }
                            });
                        });
                    }
                });
                ui.add_space(4.0);

                // MIL-SPEC registration marks for toolbar
                let toolbar_rect = ui.max_rect();
                Self::draw_corner_marks_static(ui, toolbar_rect, DIM_CYAN.gamma_multiply(0.5));
            });

        // Central panel: Visualization (inspector floats on top)
        // Note: Inspector is now drawn inside draw_visualization as floating panel
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

        // MIL-SPEC VOID BLACK background
        ui.painter().rect_filled(available_rect, 0.0, VOID_BLACK);

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

        // Draw floating inspector panel on the right
        self.draw_floating_inspector(ui, available_rect);

        // Draw MIL-SPEC corner registration marks on visualization panel
        self.draw_corner_marks(ui, available_rect, DIM_CYAN);

        // Draw loading overlay if current mode's data is not ready
        let current_mode = self.active_tab().viz_mode;
        if !self.is_mode_data_ready(current_mode) {
            self.draw_loading_overlay(ui, available_rect);
        }
    }

    /// Draw the floating inspector panel - MIL-SPEC FLOATING OVERLAY
    fn draw_floating_inspector(&mut self, ui: &mut egui::Ui, view_rect: Rect) {
        // Only show inspector when file is loaded
        if self.active_tab().file_data.is_none() {
            return;
        }

        // Calculate inspector panel rect - floating on right side with padding
        let inspector_width = 300.0;
        let padding = 12.0;
        let inspector_rect = Rect::from_min_size(
            Pos2::new(
                view_rect.max.x - inspector_width - padding,
                view_rect.min.y + padding,
            ),
            Vec2::new(inspector_width, view_rect.height() - padding * 2.0 - 50.0), // Leave room for HUD
        );

        // Semi-transparent background - MIL-SPEC panel
        ui.painter()
            .rect_filled(inspector_rect, 0.0, PANEL_DARK.gamma_multiply(0.92));

        // Border stroke
        ui.painter()
            .rect_stroke(inspector_rect, 0.0, egui::Stroke::new(1.0, INTERFACE_GRAY));

        // Corner registration marks
        Self::draw_corner_marks_static(ui, inspector_rect, DIM_CYAN.gamma_multiply(0.5));

        // Create a child UI for the inspector content with proper clipping
        let content_rect = inspector_rect.shrink(8.0);
        let mut child_ui = ui.child_ui(
            content_rect,
            egui::Layout::top_down(egui::Align::LEFT),
            None,
        );

        // Set clip rect to prevent content bleeding outside panel
        child_ui.set_clip_rect(content_rect);

        // Draw inspector content
        self.draw_inspector_content(&mut child_ui);
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

        // Draw the outline as a polyline with TACTICAL CYAN glow
        if screen_points.len() >= 2 {
            // Draw outer glow
            ui.painter().add(egui::Shape::line(
                screen_points.clone(),
                egui::Stroke::new(4.0, TACTICAL_CYAN.gamma_multiply(0.3)),
            ));
            // Draw main line
            ui.painter().add(egui::Shape::line(
                screen_points,
                egui::Stroke::new(2.0, TACTICAL_CYAN),
            ));
        }

        // Draw bounding box from cached data
        let (min_x, min_y, max_x, max_y) = cache.bbox;
        if min_x < f32::MAX {
            let box_min = world_to_screen(min_x, min_y);
            let box_max = world_to_screen(max_x, max_y);
            let box_rect = Rect::from_min_max(box_min, box_max);

            // Draw bounding box - TACTICAL CYAN
            ui.painter().rect_stroke(
                box_rect,
                0.0,
                egui::Stroke::new(1.5, TACTICAL_CYAN.gamma_multiply(0.6)),
            );
        }
    }

    /// Draw the empty state prompt - MIL-SPEC STANDBY
    fn draw_empty_state(&self, ui: &mut egui::Ui, rect: Rect) {
        let center = rect.center();

        // Draw corner registration marks
        self.draw_corner_marks(ui, rect, DIM_CYAN);

        ui.painter().text(
            center - Vec2::new(0.0, 30.0),
            egui::Align2::CENTER_CENTER,
            "[ AWAITING TARGET ]",
            egui::FontId::monospace(14.0),
            MUTED_TEXT,
        );
        ui.painter().text(
            center,
            egui::Align2::CENTER_CENTER,
            "DROP BINARY FILE TO ANALYZE",
            egui::FontId::monospace(12.0),
            DIM_CYAN,
        );
        ui.painter().text(
            center + Vec2::new(0.0, 24.0),
            egui::Align2::CENTER_CENTER,
            "STATUS: [STANDBY]",
            egui::FontId::monospace(10.0),
            MUTED_TEXT,
        );
    }

    /// Draw corner registration marks for MIL-SPEC aesthetic (static version for use in closures)
    fn draw_corner_marks_static(ui: &mut egui::Ui, rect: Rect, color: Color32) {
        let mark_size = 12.0;
        let stroke = egui::Stroke::new(1.0, color);

        // Top-left
        ui.painter().line_segment(
            [
                rect.min + Vec2::new(4.0, 4.0),
                rect.min + Vec2::new(4.0 + mark_size, 4.0),
            ],
            stroke,
        );
        ui.painter().line_segment(
            [
                rect.min + Vec2::new(4.0, 4.0),
                rect.min + Vec2::new(4.0, 4.0 + mark_size),
            ],
            stroke,
        );

        // Top-right
        ui.painter().line_segment(
            [
                Pos2::new(rect.max.x - 4.0, rect.min.y + 4.0),
                Pos2::new(rect.max.x - 4.0 - mark_size, rect.min.y + 4.0),
            ],
            stroke,
        );
        ui.painter().line_segment(
            [
                Pos2::new(rect.max.x - 4.0, rect.min.y + 4.0),
                Pos2::new(rect.max.x - 4.0, rect.min.y + 4.0 + mark_size),
            ],
            stroke,
        );

        // Bottom-left
        ui.painter().line_segment(
            [
                Pos2::new(rect.min.x + 4.0, rect.max.y - 4.0),
                Pos2::new(rect.min.x + 4.0 + mark_size, rect.max.y - 4.0),
            ],
            stroke,
        );
        ui.painter().line_segment(
            [
                Pos2::new(rect.min.x + 4.0, rect.max.y - 4.0),
                Pos2::new(rect.min.x + 4.0, rect.max.y - 4.0 - mark_size),
            ],
            stroke,
        );

        // Bottom-right
        ui.painter().line_segment(
            [
                Pos2::new(rect.max.x - 4.0, rect.max.y - 4.0),
                Pos2::new(rect.max.x - 4.0 - mark_size, rect.max.y - 4.0),
            ],
            stroke,
        );
        ui.painter().line_segment(
            [
                Pos2::new(rect.max.x - 4.0, rect.max.y - 4.0),
                Pos2::new(rect.max.x - 4.0, rect.max.y - 4.0 - mark_size),
            ],
            stroke,
        );
    }

    /// Draw corner registration marks for MIL-SPEC aesthetic
    fn draw_corner_marks(&self, ui: &mut egui::Ui, rect: Rect, color: Color32) {
        let mark_size = 20.0;
        let stroke = egui::Stroke::new(1.5, color);

        // Top-left ◢
        ui.painter().line_segment(
            [
                rect.min + Vec2::new(8.0, 8.0),
                rect.min + Vec2::new(8.0 + mark_size, 8.0),
            ],
            stroke,
        );
        ui.painter().line_segment(
            [
                rect.min + Vec2::new(8.0, 8.0),
                rect.min + Vec2::new(8.0, 8.0 + mark_size),
            ],
            stroke,
        );

        // Top-right ◣
        ui.painter().line_segment(
            [
                Pos2::new(rect.max.x - 8.0, rect.min.y + 8.0),
                Pos2::new(rect.max.x - 8.0 - mark_size, rect.min.y + 8.0),
            ],
            stroke,
        );
        ui.painter().line_segment(
            [
                Pos2::new(rect.max.x - 8.0, rect.min.y + 8.0),
                Pos2::new(rect.max.x - 8.0, rect.min.y + 8.0 + mark_size),
            ],
            stroke,
        );

        // Bottom-left
        ui.painter().line_segment(
            [
                Pos2::new(rect.min.x + 8.0, rect.max.y - 8.0),
                Pos2::new(rect.min.x + 8.0 + mark_size, rect.max.y - 8.0),
            ],
            stroke,
        );
        ui.painter().line_segment(
            [
                Pos2::new(rect.min.x + 8.0, rect.max.y - 8.0),
                Pos2::new(rect.min.x + 8.0, rect.max.y - 8.0 - mark_size),
            ],
            stroke,
        );

        // Bottom-right
        ui.painter().line_segment(
            [
                Pos2::new(rect.max.x - 8.0, rect.max.y - 8.0),
                Pos2::new(rect.max.x - 8.0 - mark_size, rect.max.y - 8.0),
            ],
            stroke,
        );
        ui.painter().line_segment(
            [
                Pos2::new(rect.max.x - 8.0, rect.max.y - 8.0),
                Pos2::new(rect.max.x - 8.0, rect.max.y - 8.0 - mark_size),
            ],
            stroke,
        );
    }

    /// Draw loading overlay when current mode's data is still computing - MIL-SPEC
    fn draw_loading_overlay(&self, ui: &mut egui::Ui, rect: Rect) {
        let tab = self.active_tab();

        // Semi-transparent dark overlay using MIL-SPEC void black
        ui.painter()
            .rect_filled(rect, 0.0, VOID_BLACK.gamma_multiply(0.85));

        // Draw corner marks
        self.draw_corner_marks(ui, rect, TACTICAL_CYAN.gamma_multiply(0.5));

        // Loading message box - MIL-SPEC panel
        let box_size = Vec2::new(300.0, 110.0);
        let box_rect = Rect::from_center_size(rect.center(), box_size);

        ui.painter().rect_filled(box_rect, 0.0, PANEL_DARK);
        ui.painter()
            .rect_stroke(box_rect, 0.0, egui::Stroke::new(2.0, TACTICAL_CYAN));

        // Mode name - MIL-SPEC format
        let mode_name = tab.viz_mode.name().to_uppercase();
        ui.painter().text(
            box_rect.center() - Vec2::new(0.0, 28.0),
            egui::Align2::CENTER_CENTER,
            format!("[ COMPUTING {} ]", mode_name),
            egui::FontId::monospace(12.0),
            DATA_WHITE,
        );

        // Status indicator
        let time = ui.ctx().input(|i| i.time);
        let scan_char = match (time * 8.0) as i32 % 4 {
            0 => "▓▓▓░░░░░",
            1 => "░▓▓▓░░░░",
            2 => "░░▓▓▓░░░",
            _ => "░░░▓▓▓░░",
        };
        ui.painter().text(
            box_rect.center() - Vec2::new(0.0, 8.0),
            egui::Align2::CENTER_CENTER,
            scan_char,
            egui::FontId::monospace(12.0),
            TACTICAL_CYAN,
        );

        // Progress hint with percentages
        if let Some(ref tasks) = tab.background_tasks {
            // Get the progress for the current mode's relevant task
            let (task_name, progress) = match tab.viz_mode {
                VisualizationMode::KolmogorovComplexity => {
                    ("COMPLEXITY", tasks.complexity_progress)
                }
                VisualizationMode::MultiScaleEntropy => ("RCMSE", tasks.rcmse_progress),
                VisualizationMode::WaveletEntropy => ("WAVELET", tasks.wavelet_progress),
                _ => ("ANALYSIS", 0.0),
            };

            let pct = (progress * 100.0) as u32;
            ui.painter().text(
                box_rect.center() + Vec2::new(0.0, 14.0),
                egui::Align2::CENTER_CENTER,
                format!("{}: {:03}%", task_name, pct),
                egui::FontId::monospace(11.0),
                CAUTION_AMBER,
            );

            // Progress bar - MIL-SPEC style
            let bar_width = 220.0;
            let bar_height = 4.0;
            let bar_rect = Rect::from_center_size(
                box_rect.center() + Vec2::new(0.0, 36.0),
                Vec2::new(bar_width, bar_height),
            );

            // Background
            ui.painter().rect_filled(bar_rect, 0.0, INTERFACE_GRAY);

            // Progress fill
            let fill_width = bar_width * progress;
            let fill_rect = Rect::from_min_size(bar_rect.min, Vec2::new(fill_width, bar_height));
            ui.painter().rect_filled(fill_rect, 0.0, TACTICAL_CYAN);
        }
    }

    /// Draw a compact progress indicator for a background task - MIL-SPEC
    fn draw_task_progress(&self, ui: &mut egui::Ui, name: &str, progress: f32) {
        let pct = (progress * 100.0) as u32;
        ui.label(
            RichText::new(format!("{}: {:03}%", name.to_uppercase(), pct))
                .size(9.0)
                .color(if progress >= 1.0 {
                    OPERATIONAL_GREEN
                } else {
                    CAUTION_AMBER
                }),
        );
    }

    /// Draw the drop target indicator - MIL-SPEC TARGET ACQUISITION
    fn draw_drop_indicator(&self, ui: &mut egui::Ui, rect: Rect) {
        // Request continuous repainting for pulse animation
        ui.ctx().request_repaint();

        // Calculate pulse intensity based on time (0.5 Hz oscillation)
        let time = ui.input(|i| i.time);
        let pulse = ((time * std::f64::consts::PI).sin() * 0.5 + 0.5) as f32;
        let pulse_color = TACTICAL_CYAN.gamma_multiply(0.5 + pulse * 0.5);

        // Darken background with tactical overlay
        ui.painter()
            .rect_filled(rect, 0.0, VOID_BLACK.gamma_multiply(0.85));

        // Tactical grid pattern
        let grid_spacing = 40.0;
        let grid_color = INTERFACE_GRAY.gamma_multiply(0.4);
        let grid_stroke = egui::Stroke::new(0.5, grid_color);

        // Vertical grid lines
        let mut x = rect.left() + grid_spacing;
        while x < rect.right() {
            ui.painter().line_segment(
                [egui::pos2(x, rect.top()), egui::pos2(x, rect.bottom())],
                grid_stroke,
            );
            x += grid_spacing;
        }

        // Horizontal grid lines
        let mut y = rect.top() + grid_spacing;
        while y < rect.bottom() {
            ui.painter().line_segment(
                [egui::pos2(rect.left(), y), egui::pos2(rect.right(), y)],
                grid_stroke,
            );
            y += grid_spacing;
        }

        // Pulsing outer border
        ui.painter().rect_stroke(
            rect.shrink(2.0),
            0.0,
            egui::Stroke::new(2.0 + pulse, pulse_color),
        );

        // Draw corner marks with pulse
        self.draw_corner_marks(ui, rect, pulse_color);

        // Target acquisition box
        let inner_rect = Rect::from_center_size(rect.center(), Vec2::new(320.0, 180.0));
        ui.painter()
            .rect_stroke(inner_rect, 0.0, egui::Stroke::new(2.0 + pulse, pulse_color));

        // Inner glow line
        let glow_rect = inner_rect.shrink(4.0);
        ui.painter().rect_stroke(
            glow_rect,
            0.0,
            egui::Stroke::new(1.0, TACTICAL_CYAN.gamma_multiply(0.3)),
        );

        // Crosshair markers at corners of target box
        let corner_size = 12.0;
        let corners = [
            inner_rect.left_top(),
            inner_rect.right_top(),
            inner_rect.left_bottom(),
            inner_rect.right_bottom(),
        ];
        for corner in corners {
            let is_left = corner.x == inner_rect.left();
            let is_top = corner.y == inner_rect.top();
            let dx = if is_left { corner_size } else { -corner_size };
            let dy = if is_top { corner_size } else { -corner_size };
            ui.painter().line_segment(
                [corner, egui::pos2(corner.x + dx, corner.y)],
                egui::Stroke::new(2.0, pulse_color),
            );
            ui.painter().line_segment(
                [corner, egui::pos2(corner.x, corner.y + dy)],
                egui::Stroke::new(2.0, pulse_color),
            );
        }

        // Header text
        ui.painter().text(
            rect.center() - Vec2::new(0.0, 50.0),
            egui::Align2::CENTER_CENTER,
            "[ TARGET ACQUISITION ]",
            egui::FontId::monospace(11.0),
            MUTED_TEXT,
        );

        // Main text with pulse
        ui.painter().text(
            rect.center(),
            egui::Align2::CENTER_CENTER,
            "DROP BINARY FILE",
            egui::FontId::monospace(20.0),
            pulse_color,
        );

        // Status text
        ui.painter().text(
            rect.center() + Vec2::new(0.0, 40.0),
            egui::Align2::CENTER_CENTER,
            "STATUS: [AWAITING TARGET]",
            egui::FontId::monospace(10.0),
            OPERATIONAL_GREEN.gamma_multiply(0.5 + pulse * 0.5),
        );
    }

    /// Draw the HUD overlay with zoom, position, and file size - MIL-SPEC FORMAT
    fn draw_hud(&self, ui: &mut egui::Ui, rect: Rect) {
        let tab = self.active_tab();
        let Some(file) = &tab.file_data else {
            return;
        };

        let hud_rect = Rect::from_min_size(
            rect.min + Vec2::new(12.0, rect.height() - 46.0),
            Vec2::new(420.0, 34.0),
        );

        // MIL-SPEC panel background
        ui.painter()
            .rect_filled(hud_rect, 2.0, PANEL_DARK.gamma_multiply(0.9));
        ui.painter()
            .rect_stroke(hud_rect, 2.0, egui::Stroke::new(1.0, INTERFACE_GRAY));

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

        // MIL-SPEC coordinate format
        let text = format!(
            "◢ [{}] | ZOOM: {:.2}x | POS: {:.0},{:.0} | SIZE: {} | [ACTIVE]",
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
            egui::FontId::monospace(10.0),
            TACTICAL_CYAN,
        );
    }

    /// Draw the inspector content (for use in floating panel) - MIL-SPEC
    fn draw_inspector_content(&mut self, ui: &mut egui::Ui) {
        if self.active_tab().file_data.is_none() {
            return;
        }

        // Wrap in ScrollArea for scrollable content
        egui::ScrollArea::vertical()
            .auto_shrink([false, false])
            .show(ui, |ui| {
                // Get file data for hex view
                let tab = &self.tabs[self.active_tab];
                let file = tab.file_data.as_ref().unwrap();
                let file_size = file.size;
                let data = Arc::clone(&file.data);
                let file_type = file.file_type;
                let selection_offset = tab.selection.offset;

                // Header: File type and size - MIL-SPEC format
                ui.horizontal(|ui| {
                    ui.label(
                        RichText::new(format!("[ {} ]", file_type.to_uppercase()))
                            .monospace()
                            .strong()
                            .color(TACTICAL_CYAN),
                    );
                    ui.label(
                        RichText::new(format!("// {}", format_bytes(file_size)))
                            .monospace()
                            .color(MUTED_TEXT),
                    );
                });
                ui.add_space(4.0);
                ui.separator();
                ui.add_space(4.0);

                // Hex View - dynamically calculate bytes per row based on width
                let panel_width = ui.available_width();
                // Calculate bytes_per_row based on available width
                // Approximate: offset(70) + gap(6) + hex bytes + gap(4) + ascii
                // For floating panel (~280px), use 8 bytes max
                let bytes_per_row = if panel_width >= 500.0 {
                    16
                } else if panel_width >= 400.0 {
                    12
                } else if panel_width >= 320.0 {
                    8
                } else {
                    4 // Narrow floating panel
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

                // Hex View with proper clipping via ScrollArea - MIL-SPEC
                egui::Frame::none()
                    .fill(VOID_BLACK)
                    .rounding(0.0)
                    .stroke(egui::Stroke::new(1.0, INTERFACE_GRAY))
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
                                        // Offset column - MIL-SPEC
                                        let offset_color = if is_selected_row {
                                            TACTICAL_CYAN
                                        } else {
                                            DIM_CYAN
                                        };
                                        ui.label(
                                            RichText::new(format!("{:08X}", row_offset))
                                                .monospace()
                                                .size(11.0)
                                                .color(offset_color),
                                        );

                                        ui.add_space(6.0);

                                        // Hex bytes - first half - MIL-SPEC
                                        for (i, &byte) in
                                            row_bytes.iter().take(half_bytes).enumerate()
                                        {
                                            let byte_offset = row_offset + i;
                                            let is_cursor_byte =
                                                byte_offset == selection_offset as usize;

                                            if is_cursor_byte {
                                                egui::Frame::none().fill(TACTICAL_CYAN).show(
                                                    ui,
                                                    |ui| {
                                                        ui.label(
                                                            RichText::new(format!("{:02X}", byte))
                                                                .monospace()
                                                                .size(11.0)
                                                                .color(VOID_BLACK),
                                                        );
                                                    },
                                                );
                                            } else {
                                                ui.label(
                                                    RichText::new(format!("{:02X}", byte))
                                                        .monospace()
                                                        .size(11.0)
                                                        .color(Self::byte_color(byte)),
                                                );
                                            }
                                        }

                                        ui.add_space(4.0);

                                        // Hex bytes - second half - MIL-SPEC
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
                                                egui::Frame::none().fill(TACTICAL_CYAN).show(
                                                    ui,
                                                    |ui| {
                                                        ui.label(
                                                            RichText::new(format!("{:02X}", byte))
                                                                .monospace()
                                                                .size(11.0)
                                                                .color(VOID_BLACK),
                                                        );
                                                    },
                                                );
                                            } else {
                                                ui.label(
                                                    RichText::new(format!("{:02X}", byte))
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

                                        // ASCII column - MIL-SPEC
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
                                                        .color(VOID_BLACK)
                                                        .background_color(TACTICAL_CYAN),
                                                );
                                            } else {
                                                ui.label(
                                                    RichText::new(ch.to_string())
                                                        .monospace()
                                                        .size(11.0)
                                                        .color(
                                                            OPERATIONAL_GREEN.gamma_multiply(0.7),
                                                        ),
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
                                                .color(MUTED_TEXT)
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

                            // Entropy bar - MIL-SPEC
                            let bar_height = 6.0;
                            let (bar_rect, _) = ui.allocate_exact_size(
                                egui::Vec2::new(ui.available_width(), bar_height),
                                Sense::hover(),
                            );
                            ui.painter().rect_filled(bar_rect, 0.0, INTERFACE_GRAY);
                            let fill_width = bar_rect.width() * (selection_entropy as f32 / 8.0);
                            let fill_rect = Rect::from_min_size(
                                bar_rect.min,
                                egui::Vec2::new(fill_width, bar_height),
                            );
                            ui.painter().rect_filled(
                                fill_rect,
                                0.0,
                                Self::entropy_color(selection_entropy),
                            );

                            // Interpretation - MIL-SPEC classification
                            let interpretation = if selection_entropy < 1.0 {
                                "CLASS: UNIFORM/EMPTY"
                            } else if selection_entropy < 3.0 {
                                "CLASS: STRUCTURED (TEXT/CODE)"
                            } else if selection_entropy < 5.0 {
                                "CLASS: MIXED DATA"
                            } else if selection_entropy < 7.0 {
                                "CLASS: HIGH ENTROPY (BINARY)"
                            } else {
                                "CLASS: ENCRYPTED/COMPRESSED"
                            };
                            ui.label(
                                RichText::new(interpretation)
                                    .monospace()
                                    .small()
                                    .color(MUTED_TEXT),
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
                                                .color(MUTED_TEXT)
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

                            // Complexity bar - MIL-SPEC
                            let bar_height = 6.0;
                            let (bar_rect, _) = ui.allocate_exact_size(
                                egui::Vec2::new(ui.available_width(), bar_height),
                                Sense::hover(),
                            );
                            ui.painter().rect_filled(bar_rect, 0.0, INTERFACE_GRAY);
                            let fill_width = bar_rect.width() * selection_complexity as f32;
                            let fill_rect = Rect::from_min_size(
                                bar_rect.min,
                                egui::Vec2::new(fill_width, bar_height),
                            );
                            ui.painter().rect_filled(
                                fill_rect,
                                0.0,
                                Self::complexity_color(selection_complexity),
                            );

                            // Interpretation - MIL-SPEC classification
                            let interpretation = if selection_complexity < 0.2 {
                                "CLASS: HIGHLY COMPRESSIBLE"
                            } else if selection_complexity < 0.4 {
                                "CLASS: SIMPLE PATTERNS"
                            } else if selection_complexity < 0.6 {
                                "CLASS: STRUCTURED DATA"
                            } else if selection_complexity < 0.8 {
                                "CLASS: COMPLEX/COMPRESSED"
                            } else {
                                "CLASS: RANDOM/ENCRYPTED"
                            };
                            ui.label(
                                RichText::new(interpretation)
                                    .monospace()
                                    .small()
                                    .color(MUTED_TEXT),
                            );
                        });

                        ui.separator();

                        // Wavelet Entropy Analysis
                        Self::section(ui, "WAVELET ENTROPY", |ui| {
                            if let Some(ref report) = wavelet_report {
                                // Wavelet decomposition info
                                ui.label(
                                    RichText::new(format!(
                                        "LEVELS: {} // CHUNKS: {}",
                                        report.num_wavelet_levels, report.num_entropy_chunks
                                    ))
                                    .monospace()
                                    .small()
                                    .color(MUTED_TEXT),
                                );

                                // Energy distribution across scales
                                ui.add_space(4.0);
                                ui.label(
                                    RichText::new("ENERGY DISTRIBUTION")
                                        .monospace()
                                        .small()
                                        .color(MUTED_TEXT),
                                );

                                // Show energy at different scales
                                let energy_spectrum = &report.energy_spectrum;
                                if !energy_spectrum.is_empty() {
                                    let total: f64 = energy_spectrum.iter().sum();
                                    if total > 0.0 {
                                        // Coarse scales (low freq) vs fine scales (high freq)
                                        let mid = energy_spectrum.len() / 2;
                                        let coarse: f64 = energy_spectrum[..mid].iter().sum();
                                        let fine: f64 = energy_spectrum[mid..].iter().sum();
                                        let coarse_ratio = coarse / total;
                                        let fine_ratio = fine / total;

                                        ui.horizontal(|ui| {
                                            ui.label(
                                                RichText::new(format!(
                                                    "LOW FREQ: {:.1}%",
                                                    coarse_ratio * 100.0
                                                ))
                                                .monospace()
                                                .small()
                                                .color(ALERT_RED.gamma_multiply(0.8)),
                                            );
                                            ui.add_space(8.0);
                                            ui.label(
                                                RichText::new(format!(
                                                    "HIGH FREQ: {:.1}%",
                                                    fine_ratio * 100.0
                                                ))
                                                .monospace()
                                                .small()
                                                .color(TACTICAL_CYAN.gamma_multiply(0.8)),
                                            );
                                        });
                                    }
                                }
                            } else {
                                ui.label(
                                    RichText::new("[ANALYZING...]")
                                        .monospace()
                                        .color(TACTICAL_CYAN),
                                );
                            }
                        });

                        // String Preview (if ASCII found) - MIL-SPEC
                        if let Some(ref ascii) = selection_ascii {
                            ui.separator();
                            Self::section(ui, "STRING EXTRACTION", |ui| {
                                egui::Frame::none()
                                    .fill(VOID_BLACK)
                                    .stroke(egui::Stroke::new(1.0, INTERFACE_GRAY))
                                    .rounding(0.0)
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
                                                .color(CAUTION_AMBER),
                                        );
                                    });
                            });
                        }
                    });
            });
    }

    /// Draw a section with a title - MIL-SPEC bracketed header with box drawing.
    fn section(ui: &mut egui::Ui, title: &str, content: impl FnOnce(&mut egui::Ui)) {
        ui.vertical(|ui| {
            // MIL-SPEC section header with box drawing characters
            ui.horizontal(|ui| {
                ui.label(
                    RichText::new("╔═══")
                        .monospace()
                        .size(10.0)
                        .color(INTERFACE_GRAY),
                );
                ui.label(
                    RichText::new(format!("[ {} ]", title))
                        .monospace()
                        .size(10.0)
                        .strong()
                        .color(TACTICAL_CYAN),
                );
                // Fill remaining width with box chars
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    ui.label(
                        RichText::new("═══╗")
                            .monospace()
                            .size(10.0)
                            .color(INTERFACE_GRAY),
                    );
                });
            });
            ui.add_space(2.0);
            // Double-line separator
            let rect = ui.available_rect_before_wrap();
            ui.painter().line_segment(
                [
                    egui::pos2(rect.left(), rect.top()),
                    egui::pos2(rect.right(), rect.top()),
                ],
                egui::Stroke::new(1.5, DIM_CYAN),
            );
            ui.add_space(6.0);
            content(ui);
            // Bottom separator
            ui.add_space(4.0);
            let rect = ui.available_rect_before_wrap();
            ui.painter().line_segment(
                [
                    egui::pos2(rect.left(), rect.top()),
                    egui::pos2(rect.right(), rect.top()),
                ],
                egui::Stroke::new(1.0, INTERFACE_GRAY),
            );
        });
        ui.add_space(8.0);
    }

    /// Draw an info row with label and value - MIL-SPEC dotted format.
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
                            .size(10.0)
                            .color(MUTED_TEXT),
                    );
                },
            );
            ui.allocate_ui_with_layout(
                egui::Vec2::new(available * 0.5, ui.spacing().interact_size.y),
                egui::Layout::right_to_left(egui::Align::Center),
                |ui| {
                    ui.label(
                        RichText::new(value)
                            .monospace()
                            .size(11.0)
                            .color(DATA_WHITE),
                    );
                },
            );
        });
    }

    /// Get color for a byte value based on its characteristics - MIL-SPEC.
    fn byte_color(byte: u8) -> Color32 {
        if byte == 0 {
            Color32::from_rgb(60, 70, 80) // Null - dim interface gray
        } else if (0x20..=0x7e).contains(&byte) {
            DATA_WHITE // Printable ASCII - data white
        } else if byte == 0xff {
            ALERT_RED // 0xFF - alert red
        } else if byte > 0x7f {
            CAUTION_AMBER // High bytes - caution amber
        } else {
            DIM_CYAN // Control chars - dim cyan
        }
    }

    /// Get color for entropy value - MIL-SPEC status colors.
    fn entropy_color(entropy: f64) -> Color32 {
        if entropy > 7.0 {
            ALERT_RED // Critical - encrypted/random
        } else if entropy > 5.0 {
            CAUTION_AMBER // Warning - compressed
        } else if entropy > 3.0 {
            TACTICAL_CYAN // Nominal - code/data
        } else {
            OPERATIONAL_GREEN // Clean - structured
        }
    }

    /// Get color for Kolmogorov complexity value (0-1) - MIL-SPEC tactical gradient.
    fn complexity_color(complexity: f64) -> Color32 {
        let t = complexity.clamp(0.0, 1.0);

        // MIL-SPEC: operational green -> dim cyan -> tactical cyan -> amber -> alert red
        let (r, g, b) = if t < 0.25 {
            // Operational green to dim cyan (simple/compressible)
            let s = t / 0.25;
            (0.0, 1.0 - s * 0.45, 0.4 + s * 0.23)
        } else if t < 0.5 {
            // Dim cyan to tactical cyan (moderate complexity)
            let s = (t - 0.25) / 0.25;
            (0.0, 0.55 + s * 0.39, 0.63 + s * 0.37)
        } else if t < 0.75 {
            // Tactical cyan to caution amber (complex)
            let s = (t - 0.5) / 0.25;
            (s * 1.0, 0.94 - s * 0.22, 1.0 - s * 1.0)
        } else {
            // Caution amber to alert red (random/encrypted)
            let s = (t - 0.75) / 0.25;
            (1.0, 0.72 - s * 0.52, s * 0.2)
        };

        Color32::from_rgb((r * 255.0) as u8, (g * 255.0) as u8, (b * 255.0) as u8)
    }

    /// Draw the help popup content - MIL-SPEC document styling.
    fn draw_help(&mut self, ui: &mut egui::Ui) {
        ui.set_min_width(380.0);

        egui::Frame::none()
            .fill(PANEL_DARK)
            .inner_margin(16.0)
            .show(ui, |ui| {
                // Classification header
                ui.horizontal(|ui| {
                    ui.label(
                        RichText::new("◢")
                            .monospace()
                            .size(10.0)
                            .color(TACTICAL_CYAN),
                    );
                    ui.label(
                        RichText::new("APEIRON // OPERATIONS MANUAL")
                            .monospace()
                            .size(11.0)
                            .strong()
                            .color(DATA_WHITE),
                    );
                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        ui.label(
                            RichText::new("◣")
                                .monospace()
                                .size(10.0)
                                .color(TACTICAL_CYAN),
                        );
                    });
                });
                ui.label(
                    RichText::new("CLASSIFICATION: UNCLASSIFIED // REV.03")
                        .monospace()
                        .size(9.0)
                        .color(MUTED_TEXT),
                );
                ui.add_space(12.0);

                // Separator
                let rect = ui.available_rect_before_wrap();
                ui.painter().line_segment(
                    [
                        egui::pos2(rect.left(), rect.top()),
                        egui::pos2(rect.right(), rect.top()),
                    ],
                    egui::Stroke::new(1.0, INTERFACE_GRAY),
                );
                ui.add_space(12.0);

                // CONTROLS SECTION
                ui.label(
                    RichText::new("[ CONTROLS ]")
                        .monospace()
                        .size(10.0)
                        .strong()
                        .color(TACTICAL_CYAN),
                );
                ui.add_space(6.0);

                Self::control_row(ui, "SCROLL", "Zoom in/out");
                Self::control_row(ui, "DRAG", "Pan view");
                Self::control_row(ui, "HOVER", "Inspect bytes");

                ui.add_space(12.0);

                // VISUALIZATION MODES SECTION
                ui.label(
                    RichText::new("[ VISUALIZATION MODES ]")
                        .monospace()
                        .size(10.0)
                        .strong()
                        .color(TACTICAL_CYAN),
                );
                ui.add_space(6.0);

                // Mode entries with tactical styling
                Self::mode_entry(
                    ui,
                    "HIL",
                    "HILBERT CURVE",
                    "Space-filling locality-preserving map",
                );
                Self::mode_entry(
                    ui,
                    "SIM",
                    "SIMILARITY MATRIX",
                    "Recurrence plot - diagonal = patterns",
                );
                Self::mode_entry(ui, "DIG", "BYTE DIGRAPH", "256x256 transition frequencies");
                Self::mode_entry(
                    ui,
                    "BPS",
                    "BYTE PHASE SPACE",
                    "Trajectory through byte space",
                );
                Self::mode_entry(
                    ui,
                    "KOL",
                    "KOLMOGOROV",
                    "Algorithmic complexity via compression",
                );
                Self::mode_entry(ui, "JSD", "JS DIVERGENCE", "Distribution anomaly detection");
                Self::mode_entry(
                    ui,
                    "MSE",
                    "MULTI-SCALE ENTROPY",
                    "RCMSE complexity analysis",
                );
                Self::mode_entry(
                    ui,
                    "WAV",
                    "WAVELET ENTROPY",
                    "Frequency-scale entropy decomposition",
                );

                ui.add_space(12.0);

                // ENTROPY LEGEND SECTION
                ui.label(
                    RichText::new("[ ENTROPY SCALE ]")
                        .monospace()
                        .size(10.0)
                        .strong()
                        .color(TACTICAL_CYAN),
                );
                ui.add_space(6.0);

                Self::legend_row(ui, DIM_CYAN, "0-2 BITS: STRUCTURED/PADDING");
                Self::legend_row(ui, TACTICAL_CYAN, "2-5 BITS: CODE/TEXT DATA");
                Self::legend_row(ui, CAUTION_AMBER, "5-7 BITS: COMPRESSED");
                Self::legend_row(ui, ALERT_RED, "7-8 BITS: ENCRYPTED/RANDOM");

                ui.add_space(12.0);

                // COMPLEXITY LEGEND SECTION
                ui.label(
                    RichText::new("[ COMPLEXITY SCALE ]")
                        .monospace()
                        .size(10.0)
                        .strong()
                        .color(TACTICAL_CYAN),
                );
                ui.add_space(6.0);

                Self::legend_row(ui, OPERATIONAL_GREEN, "LOW: HIGHLY COMPRESSIBLE");
                Self::legend_row(ui, DIM_CYAN, "MED-LOW: SIMPLE PATTERNS");
                Self::legend_row(ui, TACTICAL_CYAN, "MEDIUM: STRUCTURED DATA");
                Self::legend_row(ui, CAUTION_AMBER, "HIGH: COMPLEX/COMPRESSED");
                Self::legend_row(ui, ALERT_RED, "CRITICAL: RANDOM/ENCRYPTED");

                ui.add_space(16.0);

                // Close button with tactical styling
                ui.horizontal(|ui| {
                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        let close_btn = ui.add(
                            egui::Button::new(
                                RichText::new("[ DISMISS ]")
                                    .monospace()
                                    .size(10.0)
                                    .color(DATA_WHITE),
                            )
                            .fill(INTERFACE_GRAY)
                            .rounding(0.0),
                        );
                        if close_btn.clicked() {
                            self.show_help = false;
                        }
                    });
                });

                // Footer classification
                ui.add_space(8.0);
                ui.label(
                    RichText::new("◤ DOC.REV.03 // APEIRON-BINARY-ANALYSIS ◥")
                        .monospace()
                        .size(8.0)
                        .color(MUTED_TEXT),
                );

                // MIL-SPEC registration marks for help dialog
                let help_rect = ui.max_rect();
                Self::draw_corner_marks_static(ui, help_rect, DIM_CYAN.gamma_multiply(0.5));
            });
    }

    /// Draw a mode entry for help - MIL-SPEC format.
    fn mode_entry(ui: &mut egui::Ui, code: &str, name: &str, desc: &str) {
        ui.horizontal(|ui| {
            ui.label(
                RichText::new(format!("[{}]", code))
                    .monospace()
                    .size(10.0)
                    .color(TACTICAL_CYAN),
            );
            ui.label(
                RichText::new(name)
                    .monospace()
                    .size(10.0)
                    .strong()
                    .color(DATA_WHITE),
            );
        });
        ui.label(
            RichText::new(format!("    {}", desc))
                .monospace()
                .size(9.0)
                .color(MUTED_TEXT),
        );
        ui.add_space(4.0);
    }

    /// Draw a control hint row - MIL-SPEC format.
    fn control_row(ui: &mut egui::Ui, action: &str, description: &str) {
        ui.horizontal(|ui| {
            ui.label(
                RichText::new(action)
                    .monospace()
                    .size(10.0)
                    .strong()
                    .color(DATA_WHITE),
            );
            ui.label(
                RichText::new(format!("... {}", description))
                    .monospace()
                    .size(10.0)
                    .color(MUTED_TEXT),
            );
        });
    }

    /// Draw a color legend row - MIL-SPEC tactical format.
    fn legend_row(ui: &mut egui::Ui, color: Color32, label: &str) {
        ui.horizontal(|ui| {
            let (rect, _) = ui.allocate_exact_size(Vec2::new(12.0, 12.0), Sense::hover());
            // Square indicator instead of circle for tactical look
            ui.painter().rect_filled(
                egui::Rect::from_center_size(rect.center(), egui::Vec2::splat(8.0)),
                1.0,
                color,
            );
            ui.label(RichText::new(label).monospace().size(9.0).color(DATA_WHITE));
        });
    }
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
