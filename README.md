# Apeiron

A high-performance, GPU-accelerated binary entropy and complexity visualizer using Hilbert curves and advanced signal analysis techniques.

Apeiron provides interactive visual analysis of binary files through eight visualization modes, helping identify patterns, encrypted regions, compressed data, and structural anomalies in executables, firmware, and other binary formats. It includes wavelet-based malware detection inspired by academic research.

https://github.com/user-attachments/assets/c2ec6dc2-0296-4fa2-9494-8e86ccd562c2

## Features

- **Eight Visualization Modes**: Comprehensive binary analysis from entropy to wavelet decomposition
- **GPU Acceleration**: Hardware-accelerated rendering via wgpu compute shaders (WGSL)
- **Portable SIMD**: High-performance algorithms using the `wide` crate (AVX2 on x86_64, NEON on ARM64)
- **Progressive Rendering**: Large files (100MB+) render instantly with background refinement
- **Memory-Mapped I/O**: Efficient handling of multi-gigabyte files without loading into RAM
- **SSECS Malware Detection**: Wavelet-based suspiciousness scoring from academic research
- **Interactive Hex Inspector**: Synchronized hex view with Hilbert curve region highlighting
- **Real-time Analysis**: Hover over any region to see detailed byte-level analysis
- **Hilbert Curve Mapping**: Space-filling curve preserves locality - nearby bytes appear as nearby pixels
- **Pan & Zoom**: Explore large files with smooth navigation
- **Cross-Platform**: macOS (Apple Silicon & Intel), Linux, and Windows

## Visualization Modes

### Hilbert Curve (HIL)
Maps file bytes to a Hilbert curve with forensic color coding based on byte characteristics:
- **Blue**: Null bytes / padding / zeroes
- **Cyan**: ASCII text regions
- **Green**: Code / machine instructions
- **Red/Orange**: High entropy (compressed/encrypted data)

The Hilbert curve preserves spatial locality, meaning bytes that are close together in the file appear close together in the visualization.

### Similarity Matrix (SIM)
A recurrence plot from nonlinear dynamics theory. Each pixel (x,y) shows the similarity between the byte window at position x and position y:
- **Diagonal lines**: Repeating patterns or sequences
- **Vertical/horizontal lines**: Laminar states (unchanged regions)
- **Checkerboard patterns**: Periodic structures

Uses SIMD-accelerated chi-squared distance with branchless division for real-time computation.

### Byte Digraph (DIG)
A 256x256 heatmap showing byte transition frequencies. X-axis is the source byte value, Y-axis is the following byte value:
- **Bright regions**: Frequently occurring byte pairs
- **Dark regions**: Rare or absent transitions
- **Clusters**: Reveal character set usage (ASCII, Unicode, binary patterns)

Computed with parallel thread-local histograms and SIMD merge operations.

### Byte Phase Space (PHS)
Plots byte[i] vs byte[i+1] for all sequential bytes, colored by file position:
- Shows the file's "attractor" in phase space
- Reveals underlying data structure and patterns
- Position coloring shows how patterns evolve through the file

### Kolmogorov Complexity (KOL)
Approximates algorithmic complexity using DEFLATE compression ratio:
- **Purple/Blue**: Low complexity - highly compressible (nulls, repetitive data)
- **Teal/Green**: Medium complexity - structured data
- **Yellow/Orange**: High complexity - compressed or complex data
- **Red/Pink**: Maximum complexity - encrypted or truly random data

Features fast-path detection for likely-incompressible data, skipping compression for random/encrypted regions.

### Jensen-Shannon Divergence (JSD)
Measures how much each region's byte distribution diverges from the file's overall distribution:
- **Blue/Green**: Normal regions matching file's typical byte distribution
- **Yellow/Orange**: Anomalous regions with unusual byte patterns
- **Red**: Highly anomalous - encrypted, compressed, or foreign data

JSD is symmetric and bounded [0,1], making it ideal for detecting embedded or injected content.

### Multi-Scale Entropy (MSE)
Refined Composite Multi-Scale Entropy (RCMSE) analysis revealing complexity across multiple time scales:
- **Blue**: Low multi-scale complexity (simple, regular patterns)
- **Green/Yellow**: Medium complexity (structured data)
- **Orange/Red**: High complexity across scales (complex or random data)

MSE distinguishes between different types of complexity - truly random data vs. complex but structured data.

### Wavelet Entropy (WAV)
Haar wavelet decomposition analyzing entropic energy distribution across spatial scales:
- **Blue/Green**: Fine-scale energy dominance (normal variation, clean files)
- **Yellow/Orange**: Mixed energy distribution (suspicious patterns)
- **Red**: Coarse-scale energy dominance (large entropy shifts, potentially malicious)

Based on the SSECS methodology from Wojnowicz et al. (2016).

## SSECS Malware Analysis

The right panel includes **SSECS (Suspiciously Structured Entropic Change Score)** analysis based on:

> Wojnowicz et al., "Wavelet Decomposition of Software Entropy Reveals Symptoms of Malicious Code"  
> *Journal of Innovation in Digital Ecosystems* (2016)

Key insight: Malware tends to concentrate entropic energy at **coarse** spatial resolution levels (large shifts between encrypted/compressed sections), while legitimate files concentrate energy at **fine** levels (small local variations).

The analysis shows:
- **Malware Probability**: Estimated likelihood of malicious content (0-100%)
- **Classification**: Clean / Suspicious / Likely Malware
- **Energy Distribution**: Coarse vs. fine energy ratio
- **Wavelet Levels**: Number of decomposition levels analyzed

## Interactive Hex Inspector

The right panel provides a synchronized hex view that:
- Shows bytes at the current cursor position
- Highlights the visible hex region on the Hilbert curve visualization
- Displays offset in both hex and decimal
- Shows ASCII representation alongside hex values
- Scrolls through the file with the visualization

## Installation

### Pre-built Binaries
Download the latest release for your platform from the [Releases](../../releases) page:
- `apeiron-macos-arm64` - macOS Apple Silicon (M1/M2/M3)
- `apeiron-macos-x86_64` - macOS Intel
- `apeiron-linux-x86_64` - Linux x86_64
- `apeiron-windows-x86_64.exe` - Windows x86_64

### Building from Source

Requirements:
- Rust 1.70+ (install via [rustup](https://rustup.rs))
- On Linux: `libxcb`, `libxkbcommon`, `libgtk-3`

```bash
# Clone the repository
git clone https://github.com/anomalyco/apeiron.git
cd apeiron

# Build release version (with LTO optimization)
cargo build --release

# Run
./target/release/apeiron
```

#### Linux Dependencies
```bash
# Debian/Ubuntu
sudo apt-get install libxcb-render0-dev libxcb-shape0-dev libxcb-xfixes0-dev libxkbcommon-dev libgtk-3-dev

# Fedora
sudo dnf install libxcb-devel libxkbcommon-devel gtk3-devel

# Arch Linux
sudo pacman -S libxcb libxkbcommon gtk3
```

## Usage

1. **Open a file**: Drag and drop any binary file onto the window, or click "Open File..."
2. **Navigate**: 
   - Scroll to zoom in/out
   - Click and drag to pan
   - Hover over regions to inspect bytes
3. **Switch modes**: Use the Mode dropdown in the toolbar
4. **Reset view**: Click "Reset View" to fit the visualization to the window
5. **Analyze**: Review the SSECS malware analysis and entropy metrics in the right panel

## Controls

| Action | Control |
|--------|---------|
| Zoom | Scroll wheel |
| Pan | Click and drag |
| Inspect | Hover over pixels |
| Open file | Drag & drop or "Open File..." button |
| Reset view | "Reset View" button |
| Help | "Help" button |

## Data Inspector Panel

The right panel shows detailed information about the currently hovered byte position:

### File Information
- **File Type**: Auto-detected via magic bytes (PE, ELF, Mach-O, ZIP, PDF, etc.)
- **File Size**: Human-readable size

### Cursor Location
- **Offset (Hex)**: Current byte position in hexadecimal
- **Offset (Dec)**: Current byte position in decimal

### Entropy Analysis
- **Entropy**: Shannon entropy (0-8 bits) with visual bar
- **Interpretation**: Low / Medium / High entropy classification

### Kolmogorov Complexity
- **Complexity**: Compression ratio percentage
- **Interpretation**: Simple / Structured / Complex / Random

### SSECS (Wavelet Entropy)
- **Malware Probability**: Percentage likelihood
- **Classification**: Clean / Suspicious / Likely Malware
- **Energy Distribution**: Coarse vs. fine ratio
- **Levels/Chunks**: Wavelet decomposition statistics

### Hex View
- Interactive hex dump with ASCII representation
- Scrollable through entire file
- Current position highlighted
- Region outline synced with visualization

## Technical Details

### Entropy Calculation
Shannon entropy is calculated over a sliding window using SIMD-accelerated histogram counting:
```
H = -Σ p(x) * log₂(p(x))
```
where p(x) is the probability of byte value x in the window. Result ranges from 0 (uniform) to 8 bits (maximum entropy).

**Optimizations:**
- 4-way parallel histogram counting to avoid cache contention
- Cache-aligned (64-byte) histogram buffers
- True SIMD log2 approximation using IEEE 754 bit manipulation
- Dual accumulators for instruction-level parallelism

### Kolmogorov Complexity Approximation
Complexity is approximated using DEFLATE compression (level 6):
```
K(x) ≈ len(compress(x)) / len(x)
```
Pre-computed at file load time using parallel processing (sampled every 64 bytes with 128-byte windows).

**Optimizations:**
- Thread-local encoder buffer reuse
- Fast-path detection for likely-incompressible data (skips compression)
- Background streaming computation with progress updates

### Jensen-Shannon Divergence
JSD between window distribution P and file distribution Q:
```
JSD(P||Q) = ½ D_KL(P||M) + ½ D_KL(Q||M)
```
where M = ½(P + Q) and D_KL is Kullback-Leibler divergence.

**Optimizations:**
- SIMD f64x4 for 256-element distribution operations
- Fused mixture + KL computation reducing memory passes
- Dual accumulators for better ILP

### Multi-Scale Entropy (RCMSE)
Refined Composite Multi-Scale Entropy using coarse-graining:
1. Coarse-grain signal at multiple scales τ
2. Compute sample entropy at each scale
3. Average across coarse-grained sequences
4. Aggregate into complexity score

**Optimizations:**
- Direct byte processing (eliminates intermediate f64 allocation)
- SIMD coarse-graining with u32x4 accumulation
- Parallel scale computation with per-thread buffers

### Wavelet Transform (SSECS)
Orthonormal Haar wavelet decomposition following Wojnowicz et al.:

1. **Entropy Stream**: File split into 256-byte chunks, Shannon entropy computed per chunk
2. **Haar Transform**: Wavelet coefficients d_{jk} computed with 1/√2 scaling (energy-preserving)
3. **Energy Spectrum**: E_j = Σ(d_{jk})² for each resolution level j
4. **SSECS Score**: Logistic model based on coarse/fine energy ratio

**Optimizations:**
- SIMD Haar transform using f64x4
- Thread-local scratch buffers with UnsafeCell
- 8-way energy spectrum computation with dual accumulators

### Hilbert Curve
The Hilbert curve dimension is chosen as the smallest power of 2 where n² >= file_size. This ensures all bytes can be mapped while maintaining the locality-preserving property.

**Optimizations:**
- Precomputed lookup tables for dimensions 64, 128, 256, 512 (O(1) access)
- Lazy initialization with `OnceLock`
- Batch conversion functions for SIMD-friendly processing

### GPU Acceleration
When available, visualization rendering uses wgpu compute shaders (WGSL) for parallel pixel generation:
- **Hilbert**: Computes d2xy transform and byte analysis per pixel
- **Digraph**: Parallel frequency counting with atomic operations
- **Phase Space**: Trajectory accumulation with position coloring
- **Similarity Matrix**: Chi-squared distance computation

Falls back to CPU (with rayon parallelization) for modes requiring CPU-side computation (KOL, JSD, MSE, WAV) or when GPU is unavailable.

### Progressive Rendering
Files larger than 100MB use a two-phase rendering approach:
1. **Coarse pass**: ~10K hierarchical samples for instant preview
2. **Fine pass**: Full precision sequential computation in background

The main thread reads computed values lock-free while the background thread refines data progressively.

## Performance

- **Large Files**: 100MB+ files handled efficiently via viewport-aware rendering and memory-mapped I/O
- **Pre-computation**: Kolmogorov, RCMSE, and wavelet maps computed at load time using parallel processing
- **GPU Acceleration**: Significant speedup for Hilbert, Digraph, Phase Space, and Similarity Matrix modes
- **Portable SIMD**: AVX2 on x86_64, NEON on ARM64 via the `wide` crate
- **Texture Caching**: Smart regeneration thresholds prevent excessive recomputation during navigation
- **Memory Efficient**: Streaming hex view renders only visible rows; mmap for file access

## File Type Detection

Apeiron automatically detects common file types via magic bytes:

| Category | Formats |
|----------|---------|
| Executables | PE (EXE/DLL), ELF, Mach-O (all variants) |
| Archives | ZIP, RAR, GZIP, BZIP2, 7-Zip, XZ |
| Images | PNG, JPEG, GIF, BMP, TIFF |
| Documents | PDF |
| Media | MP4/MOV, WAV/AVI (RIFF), MP3 |
| Databases | SQLite |
| Other | Java CLASS, WebAssembly (WASM) |

## Use Cases

- **Malware Analysis**: Identify packed/encrypted sections, detect suspicious entropy patterns via SSECS
- **Firmware Analysis**: Find compressed regions, locate file systems, identify anomalies
- **Forensics**: Detect hidden data, identify file fragments, find injected content
- **Reverse Engineering**: Understand binary structure, locate interesting regions
- **Data Recovery**: Locate file boundaries in raw disk images
- **Security Research**: Analyze encryption patterns, study packing techniques
- **CTF Competitions**: Quickly identify steganography, hidden data, or unusual structures

## References

- Wojnowicz, M., et al. (2016). "Wavelet Decomposition of Software Entropy Reveals Symptoms of Malicious Code." *Journal of Innovation in Digital Ecosystems*, 3, 130-140. [DOI](https://doi.org/10.1016/j.jides.2016.10.009)
- Lyda, R., & Hamrock, J. (2007). "Using Entropy Analysis to Find Encrypted and Packed Malware." *IEEE Security & Privacy*, 5(2), 40-45.
- Costa, M., et al. (2002). "Multiscale entropy analysis of complex physiologic time series." *Physical Review Letters*, 89(6).
- Hilbert, D. (1891). "Über die stetige Abbildung einer Linie auf ein Flächenstück." *Mathematische Annalen*, 38, 459-460.

## License

MIT License - See [LICENSE](LICENSE) for details.

## Acknowledgments

- Inspired by binary visualization research and tools like binvis.io and Veles
- SSECS methodology from Cylance research (Wojnowicz et al.)
- Uses [egui](https://github.com/emilk/egui) for the immediate mode GUI
- GPU compute via [wgpu](https://github.com/gfx-rs/wgpu)
- Portable SIMD via [wide](https://github.com/Lokathor/wide)
- File dialogs via [rfd](https://github.com/PolyMeilex/rfd)
- Parallel processing via [rayon](https://github.com/rayon-rs/rayon)
