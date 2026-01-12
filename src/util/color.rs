//! Color utility functions for visualization.
//!
//! Consolidates all color conversion and mapping functions used throughout
//! the application, eliminating duplication between modules.

use eframe::egui::Color32;

/// Convert HSV color to RGB Color32.
///
/// # Arguments
/// * `hue` - Hue in degrees (0-360)
/// * `saturation` - Saturation (0.0-1.0)
/// * `value` - Value/brightness (0.0-1.0)
///
/// # Returns
/// An egui Color32 value
pub fn hsv_to_color32(hue: f64, saturation: f64, value: f64) -> Color32 {
    let (r, g, b) = hsv_to_rgb_f32(hue as f32, saturation as f32, value as f32);
    Color32::from_rgb((r * 255.0) as u8, (g * 255.0) as u8, (b * 255.0) as u8)
}

/// Convert HSV to RGB as floating point values (0.0-1.0).
///
/// # Arguments
/// * `h` - Hue in degrees (0-360)
/// * `s` - Saturation (0.0-1.0)
/// * `v` - Value/brightness (0.0-1.0)
///
/// # Returns
/// A tuple of (r, g, b) values in the range 0.0-1.0
#[inline]
pub fn hsv_to_rgb_f32(h: f32, s: f32, v: f32) -> (f32, f32, f32) {
    let c = v * s;
    let x = c * (1.0 - ((h / 60.0) % 2.0 - 1.0).abs());
    let m = v - c;

    let (r, g, b) = if h < 60.0 {
        (c, x, 0.0)
    } else if h < 120.0 {
        (x, c, 0.0)
    } else if h < 180.0 {
        (0.0, c, x)
    } else if h < 240.0 {
        (0.0, x, c)
    } else if h < 300.0 {
        (x, 0.0, c)
    } else {
        (c, 0.0, x)
    };

    (r + m, g + m, b + m)
}

/// Convert HSV to RGB as u8 tuple.
///
/// # Arguments
/// * `h` - Hue in degrees (0-360)
/// * `s` - Saturation (0.0-1.0)
/// * `v` - Value/brightness (0.0-1.0)
///
/// # Returns
/// A tuple of (r, g, b) values in the range 0-255
#[inline]
pub fn hsv_to_rgb(h: f32, s: f32, v: f32) -> (u8, u8, u8) {
    let (r, g, b) = hsv_to_rgb_f32(h, s, v);
    ((r * 255.0) as u8, (g * 255.0) as u8, (b * 255.0) as u8)
}

/// Map entropy value (0-8) to a perceptually uniform color.
///
/// Uses a custom colormap optimized for entropy visualization:
/// - 0-2 (low): Blue tones (structured/padding)
/// - 2-5 (medium): Green-yellow (code/data)
/// - 5-7 (high): Orange-red (compressed)
/// - 7-8 (very high): Red-white (encrypted/random)
pub fn entropy_to_color32(entropy: f64) -> Color32 {
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

/// Convert entropy value to RGB color for Hilbert rendering (full precision).
///
/// Used for progressive Hilbert rendering.
pub fn entropy_to_rgb(entropy: f32) -> (u8, u8, u8) {
    // Same color mapping as the main visualization
    let h = entropy * 0.7; // Hue: 0 (red/low entropy) to 0.7 (blue/high entropy)
    let s = 0.85;
    let v = 0.3 + entropy * 0.6; // Brighter for higher entropy

    hsv_to_rgb(h * 360.0, s, v)
}

/// Convert entropy value to RGB color (coarse/preview quality).
/// Slightly desaturated to indicate lower precision.
pub fn entropy_to_rgb_preview(entropy: f32) -> (u8, u8, u8) {
    let h = entropy * 0.7;
    let s = 0.6; // Less saturated
    let v = 0.25 + entropy * 0.5; // Slightly dimmer

    hsv_to_rgb(h * 360.0, s, v)
}

/// Generate an animated pulse color for uncomputed regions.
///
/// Creates a gentle pulsing effect between dark gray shades.
pub fn placeholder_pulse_color(time_seconds: f64) -> (u8, u8, u8) {
    // Gentle pulse: 0.5 Hz oscillation
    let pulse = ((time_seconds * std::f64::consts::PI).sin() * 0.5 + 0.5) as f32;
    let base: f32 = 25.0;
    let range: f32 = 12.0;
    let v = (base + pulse * range) as u8;
    // Slight blue tint to indicate "working"
    (v, v, v.saturating_add(8))
}

/// Map similarity value (0-1) to color for recurrence plots.
/// Uses "inferno" colormap style for better perceptual uniformity.
///
/// - 0 (dissimilar): Black/dark purple
/// - 0.5 (moderate): Red/orange
/// - 1.0 (identical): Bright yellow/white
/// - Diagonal elements get special highlighting
pub fn similarity_to_color32(similarity: f64, is_diagonal: bool) -> Color32 {
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

/// Get color for a byte value based on its characteristics.
/// Used for hex view highlighting.
pub fn byte_color(byte: u8) -> Color32 {
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

/// Get color for entropy value in the inspector panel.
pub fn entropy_indicator_color(entropy: f64) -> Color32 {
    if entropy > 7.0 {
        Color32::from_rgb(255, 80, 80) // Red
    } else if entropy > 4.0 {
        Color32::from_rgb(80, 255, 80) // Green
    } else {
        Color32::from_rgb(80, 150, 255) // Blue
    }
}

/// Get color for Kolmogorov complexity value (0-1).
/// Uses viridis-inspired colormap.
pub fn complexity_color(complexity: f64) -> Color32 {
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hsv_to_rgb() {
        // Red
        let (r, g, b) = hsv_to_rgb(0.0, 1.0, 1.0);
        assert_eq!(r, 255);
        assert!(g < 5);
        assert!(b < 5);

        // Green
        let (r, g, b) = hsv_to_rgb(120.0, 1.0, 1.0);
        assert!(r < 5);
        assert_eq!(g, 255);
        assert!(b < 5);

        // Blue
        let (r, g, b) = hsv_to_rgb(240.0, 1.0, 1.0);
        assert!(r < 5);
        assert!(g < 5);
        assert_eq!(b, 255);
    }

    #[test]
    fn test_entropy_colors() {
        let low = entropy_to_rgb(0.0);
        let high = entropy_to_rgb(1.0);
        // Low entropy should be reddish, high entropy should be bluish
        assert!(low.0 > low.2); // More red than blue
        assert!(high.2 > high.0); // More blue than red
    }

    #[test]
    fn test_placeholder_color() {
        let (r, g, b) = placeholder_pulse_color(0.0);
        assert!(r >= 20 && r <= 40);
        assert!(g >= 20 && g <= 40);
        assert!(b >= 28 && b <= 48); // Blue tint

        let (r2, g2, b2) = placeholder_pulse_color(0.5);
        // Should be different due to pulse
        assert!(r != r2 || g != g2 || b != b2);
    }
}
