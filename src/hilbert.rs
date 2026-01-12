//! Hilbert curve algorithms for mapping between 1D and 2D coordinates.
//!
//! The Hilbert curve is a space-filling curve that provides good locality preservation,
//! making it ideal for visualizing sequential binary data in 2D space.

/// Rotate/flip a quadrant appropriately for the Hilbert curve transformation.
///
/// Uses signed arithmetic internally to handle the reflection operation correctly,
/// as the subtraction `n - 1 - x` can produce negative intermediate values
/// when x >= n (which happens during xy2d when coordinates exceed the current quadrant size).
#[inline]
fn rot(n: u64, x: &mut u64, y: &mut u64, rx: u64, ry: u64) {
    if ry == 0 {
        if rx == 1 {
            // Use signed arithmetic to avoid underflow when x or y >= n
            let n_minus_1 = n as i64 - 1;
            *x = (n_minus_1 - *x as i64) as u64;
            *y = (n_minus_1 - *y as i64) as u64;
        }
        std::mem::swap(x, y);
    }
}

/// Convert (x, y) coordinates to distance along the Hilbert curve.
///
/// # Arguments
/// * `n` - The dimension of the space (must be a power of 2)
/// * `x` - X coordinate
/// * `y` - Y coordinate
///
/// # Returns
/// The distance `d` along the Hilbert curve
pub fn xy2d(n: u64, mut x: u64, mut y: u64) -> u64 {
    let mut d = 0u64;
    let mut s = n / 2;

    while s > 0 {
        let rx = u64::from((x & s) > 0);
        let ry = u64::from((y & s) > 0);
        d += s * s * ((3 * rx) ^ ry);
        rot(s, &mut x, &mut y, rx, ry);
        s /= 2;
    }

    d
}

/// Convert distance along the Hilbert curve to (x, y) coordinates.
///
/// # Arguments
/// * `n` - The dimension of the space (must be a power of 2)
/// * `d` - Distance along the Hilbert curve
///
/// # Returns
/// A tuple of (x, y) coordinates
#[allow(dead_code)]
pub fn d2xy(n: u64, d: u64) -> (u64, u64) {
    let mut x = 0u64;
    let mut y = 0u64;
    let mut s = 1u64;
    let mut t = d;

    while s < n {
        let rx = 1 & (t / 2);
        let ry = 1 & (t ^ rx);

        rot(s, &mut x, &mut y, rx, ry);

        x += s * rx;
        y += s * ry;
        t /= 4;
        s *= 2;
    }

    (x, y)
}

/// Calculate the required dimension (power of 2) to fit a given file size.
///
/// The dimension N is chosen such that N*N >= file_size and N is a power of 2.
pub fn calculate_dimension(file_size: u64) -> u64 {
    if file_size == 0 {
        return 64;
    }

    let side = (file_size as f64).sqrt();
    let mut n = 1u64;

    while (n as f64) < side {
        n *= 2;
    }

    // Minimum 64 for very small files, no maximum - scale with file size
    n.max(64)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_roundtrip() {
        let n = 256;
        for d in 0..100 {
            let (x, y) = d2xy(n, d);
            let d2 = xy2d(n, x, y);
            assert_eq!(d, d2, "Roundtrip failed for d={d}");
        }
    }

    #[test]
    fn test_dimension_calculation() {
        assert_eq!(calculate_dimension(0), 64);
        assert_eq!(calculate_dimension(1000), 64); // sqrt(1000) ≈ 32, rounds up to 64
        assert_eq!(calculate_dimension(10000), 128); // sqrt(10000) = 100, rounds up to 128
        assert!(calculate_dimension(100_000_000) >= 10000);
    }

    #[test]
    fn test_xy2d_edge_cases() {
        // Test cases that would cause overflow with unsigned subtraction
        // when x or y >= current quadrant size s during rotation
        let n = 4;

        // (3, 0) caused overflow: s=2, rx=1, ry=0, rot tries 2-1-3 = -2
        let d = xy2d(n, 3, 0);
        let (x, y) = d2xy(n, d);
        assert_eq!((x, y), (3, 0), "Roundtrip failed for (3, 0)");

        // Test all corners and edges of a 4x4 grid
        for x in 0..n {
            for y in 0..n {
                let d = xy2d(n, x, y);
                let (x2, y2) = d2xy(n, d);
                assert_eq!((x, y), (x2, y2), "Roundtrip failed for ({x}, {y})");
            }
        }
    }

    #[test]
    fn test_full_roundtrip_larger() {
        // Test complete roundtrip for larger grid to catch any edge cases
        let n = 64;
        for d in 0..(n * n) {
            let (x, y) = d2xy(n, d);
            let d2 = xy2d(n, x, y);
            assert_eq!(d, d2, "Roundtrip d->xy->d failed for d={d}");
        }
    }
}
