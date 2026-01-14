//! Application state and types for Apeiron.
//!
//! This module contains the core application state structures:
//! - `ApeironApp` - Main application state
//! - `Tab` - Individual file tab state
//! - `FileData` - Loaded file information
//! - `Viewport` - Pan/zoom state
//! - `Selection` - Current byte selection
//! - `HexView` - Hex viewer state
//! - `VisualizationMode` - Available visualization modes
//! - Background task management

mod state;
mod types;

pub use state::{ApeironApp, Tab};
pub use types::{
    BackgroundTask, BackgroundTasks, FileData, HexView, OutlineCache, Selection, TextureParams,
    Viewport, VisualizationMode, COMPLEXITY_SAMPLE_INTERVAL, RCMSE_SAMPLE_INTERVAL,
};
