# Apeiron

A high-performance, GPU-accelerated binary entropy and complexity visualizer using Hilbert curves and advanced signal analysis techniques.

Apeiron provides interactive visual analysis of binary files through seven visualization modes, helping identify patterns, encrypted regions, compressed data, and structural anomalies in executables, firmware, and other binary formats.

https://github.com/user-attachments/assets/c2ec6dc2-0296-4fa2-9494-8e86ccd562c2

## Features

- **Seven Visualization Modes**: Comprehensive binary analysis from entropy to multi-scale complexity
- **GPU Acceleration**: Hardware-accelerated rendering via wgpu compute shaders (WGSL)
- **Portable SIMD**: High-performance algorithms using the `wide` crate (AVX2 on x86_64, NEON on ARM64)
- **Progressive Rendering**: Large files (100MB+) render instantly with background refinement
- **Memory-Mapped I/O**: Efficient handling of multi-gigabyte files without loading into RAM
- **Tiered Compression**: Adaptive Kolmogorov complexity using XZ (LZMA2) and Zstd based on file size
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
Approximates algorithmic complexity using a tiered compression system that adapts to file size:
- **Purple/Blue**: Low complexity - highly compressible (nulls, repetitive data)
- **Teal/Green**: Medium complexity - structured data
- **Yellow/Orange**: High complexity - compressed or complex data
- **Red/Pink**: Maximum complexity - encrypted or truly random data

Uses XZ (LZMA2) for small/medium files (best compression ratio) and Zstd for large files (high throughput).

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

Uses an optimized histogram-based fast approximation with O(n) complexity instead of O(n^2).

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
- Rust toolchain compatible with `Cargo.lock` (the verified benchmark toolchain is recorded below; install via [rustup](https://rustup.rs))
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
5. **Analyze**: Review the entropy and complexity metrics in the right panel

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
- Exact count-logarithm lookup, with sparse updates for short windows
- Wide histogram reduction for inputs above 4 GiB

### Kolmogorov Complexity Approximation
Complexity is approximated using a tiered compression system that adapts to file/chunk size:

| Tier | Size Range | Algorithm | Throughput |
|------|------------|-----------|------------|
| Streaming | <4KB | Zstd -1 | ~300 MB/s |
| 1 | 4KB-1MB | XZ -9 | ~0.9 MB/s |
| 2 | 1-64MB | XZ -6 | ~1.1 MB/s |
| 3 | 64MB-1GB | Zstd -19 | ~1.2 MB/s |
| 4 | 1-16GB | Zstd -8 | ~36 MB/s |
| 5 | 16-100GB | Zstd -1 | ~233 MB/s |

Pre-computed on demand when switching to Kolmogorov mode (sampled every 64 bytes with 128-byte windows).

**Optimizations:**
- XZ (LZMA2) for maximum compression ratio on small/medium data
- Zstd for high throughput on large files
- Reused small-window Zstd contexts with independent streaming frames
- Bounded background batches with progress updates
- Lazy computation: only computed when user switches to KOL mode

### Jensen-Shannon Divergence
JSD between window distribution P and file distribution Q:
```
JSD(P||Q) = ½ D_KL(P||M) + ½ D_KL(Q||M)
```
where M = ½(P + Q) and D_KL is Kullback-Leibler divergence.

**Optimizations:**
- Prepared contribution tables for 64-byte windows against a fixed reference
- No logarithms in the full-window rendering loop; EOF windows use the general kernel
- One analysis per distinct byte coordinate when zoomed in
- Large-file reference sampling uses contiguous 16 KiB blocks with file-size weighting

### Multi-Scale Entropy (RCMSE)
Refined Composite Multi-Scale Entropy using a fast histogram-based approximation:
1. Compute byte histograms for pattern counting (O(n) instead of O(n²))
2. Sample sparse scales [1, 3, 6] instead of all scales 1-6
3. Aggregate into complexity score

**Optimizations:**
- Histogram-based pattern matching (~50-100x speedup)
- O(n × 3) complexity instead of O(n² × 36)
- Summed-area tables for tolerance neighborhoods, cached rounded suffix contributions
- Reused coarse-graining scratch buffers
- Lazy computation: only computed when user switches to MSE mode

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
- **Similarity Matrix**: Chi-squared distance computation with one normalized histogram per axis coordinate

Falls back to CPU (with rayon parallelization) for modes requiring CPU-side computation (KOL, JSD, MSE) or when GPU is unavailable. GPU similarity retains its existing 16-bin approximation; the CPU mode uses 256 bins. GPU output textures and readback buffers are reused at the same dimensions. Digraph and phase-space textures use their native 256-by-256 bin resolution with nearest-neighbor display sampling.

### Progressive Rendering
Hilbert mode on files larger than 100 MiB starts a two-phase rendering approach on demand:
1. **Coarse pass**: ~10K hierarchical samples for instant preview
2. **Fine pass**: Full precision sequential computation in background

The main thread reads computed values lock-free while the background thread refines data progressively.

## Performance

- **Large Files**: 100MB+ files handled efficiently via viewport-aware rendering and memory-mapped I/O
- **Lazy Computation**: Kolmogorov and RCMSE maps computed on-demand when switching to those modes
- **GPU Acceleration**: Significant speedup for Hilbert, Digraph, Phase Space, and Similarity Matrix modes
- **Portable SIMD**: AVX2 on x86_64, NEON on ARM64 via the `wide` crate
- **Texture Caching**: Smart regeneration thresholds prevent excessive recomputation during navigation
- **Memory Efficient**: Streaming hex view renders only visible rows; mmap for file access

### Reproducible performance review

The 2026-10-11 review compares the original kernels at commit `0b0715d` with the optimized implementation. Timing summaries and environment details are stored in [`benchmarks/results/2026-10-11`](benchmarks/results/2026-10-11/metadata.json). All timings use release builds (`opt-level = 3`, LTO), one excluded warmup and 15 measured repetitions. The host is an Apple M4 Max with 16 logical CPUs, running `rustc 1.100.0-nightly (f248f4038 2026-09-05)` on `aarch64-apple-darwin`.

```bash
cargo run --release --example performance -- 15
RAYON_NUM_THREADS=1 cargo run --release --example performance -- 15 jsd512
cargo run --release --example gpu_performance -- 15
```

`performance` accepts an optional workload-name filter after the repetition count. Inputs are deterministic 8 MiB arrays: a fixed-seed xorshift byte stream and a mixed stream alternating 4 KiB zero, ASCII, ramp and random regions. Entropy uses 16,384 windows, compression uses 2,048 windows, and quick RCMSE uses 256 windows. GPU measurements cover upload submission or compute plus synchronous pixel readback; egui texture upload and complete GUI frames are outside their timing scope. The GPU upload now defers transition statistics until DIG/PHS is first used. Cached GPU rendering excludes that first-use full-file scan.

Primary CPU comparison [A1, A2]: one Rayon worker, the same input and release settings, with original and optimized executions adjacent for each workload filter. Every measured case is listed; ratios below 1 indicate a slower result in this run.

| Workload | Before median (ms) | After median (ms) | Ratio |
|----------|--------------------|-------------------|-------|
| `entropy64_random` | 15.192 | 1.313 | 11.57x |
| `entropy64_mixed` | 14.924 | 1.614 | 9.25x |
| `kolmogorov128_random` | 3.252 | 0.771 | 4.22x |
| `kolmogorov128_mixed` | 4.379 | 1.799 | 2.43x |
| `rcmse256_random` | 8.348 | 1.835 | 4.55x |
| `rcmse256_mixed` | 4.108 | 1.070 | 3.84x |
| `jsd512` | 178.272 | 22.669 | 7.86x |
| `jsd512_zoom` | 313.778 | 2.931 | 107.05x |
| `similarity512_random` | 86.244 | 15.942 | 5.41x |
| `similarity512_mixed` | 76.388 | 2.091 | 36.54x |
| `linear512` | 4.862 | 5.048 | 0.96x |
| `hilbert512` | 10.827 | 12.966 | 0.84x |
| `digraph2048` | 55.347 | 15.018 | 3.69x |
| `phase2048` | 40.824 | 15.123 | 2.70x |

The parallel CPU snapshot and GPU timings are retained separately with their observed min/max. Scheduling contention in those runs prevents treating their ratios as an application-wide speed guarantee. The linear overview path is approximately unchanged to slightly slower; its additional coordinate validation is retained. CPU Hilbert overview was also slower in this run, with a wider observed timing range; these paths are not claimed as measured speedups.

The following changes address the measured hot paths:

| Path | Work removed | Remaining time and space bounds |
|------|--------------|---------------------------------|
| Entropy | Polynomial logarithms over all 256 bins in every small window | O(W) short-window time, O(256) histogram space and a shared count-log table |
| JSD | Hundreds of logarithms per pixel | O(256W) preparation, O(W + U) full-window time; EOF uses the general kernel |
| CPU similarity | Two histograms per pixel and dense 256-bin comparison | O(TW) axis preparation, O(T^2 U) comparisons and O(256T) axis storage |
| GPU similarity | Private per-pixel histograms and repeated file-byte reads | O(TW) axis preparation and O(16T^2) comparisons, retaining the existing 16-bin approximation |
| Background results | Repeated copies of every previously computed sample | O(N) total transfer; eight queued batches of at most 16,384 `f32` values |
| Pair views | Repeated colors and an 8x oversampled texture on each axis | 256-by-256 native bins; 64x fewer output pixels and RGBA bytes |

Here W is window length in bytes, U is the number of occupied byte bins, T is square texture width, and N is the number of background samples. The native bin reduction changes each RGBA output from 16,777,216 bytes to 262,144 bytes. For a 1 GiB file at a 64-byte stride, N = 2^30 / 64 = 16,777,216 samples. The old 100,000-sample cumulative snapshots transferred 5,678,308,864 bytes; the new batches transfer N * 4 bytes = 67,108,864 bytes, an 84.6x reduction in transferred values. This is a calculation from the batching algorithms, distinct from measured latency.

Large-file JSD reference sampling reads at most 9,994,240 body bytes in 610 contiguous 16 KiB blocks, plus the first MiB. Header and body distributions receive their actual file-size weights. This replaces millions of scattered byte reads and corrects the previous header overweighting. Sampling is still an estimate and can miss narrow or rare regions.

Correctness coverage compares entropy and prepared JSD with scalar calculations to an absolute tolerance of 1e-12 in the tested cases, compares quick RCMSE histogram counts exactly with the prior approximation, and checks independent Zstd frame sizes and decompression. Additional tests cover partial final windows, cross-chunk transitions, concurrent mode completion, cancellation, texture reuse, final refinement publication, GPU trailing words and active-tab GPU identity. Native GPU similarity colors are checked within two 8-bit channel levels against the same 16-bin scalar definition.

| Assumption | Dependent conclusions | Stress test or falsification probe |
|------------|-----------------------|-----------------------------------|
| A1: Synthetic inputs exercise relevant hot paths | Timing comparisons | Random and mixed data, overview and zoom; real-file and complete-frame applicability remains unknown |
| A2: Release settings and worker count are held fixed within each comparison | Speed ratios | Record compiler, profile and worker count; keep observed min/max and separate the contended 16-worker run |
| A3: Display coordinates and reference distributions follow application contracts | Numerical and pixel equivalence | Negative coordinates, EOF, zero texture size, sparse and uniform references, partial windows and scalar comparisons |
| A4: Native validation on this macOS host covers this GPU path | GPU correctness and timing | Adapter-required tests executed; Windows and Linux native performance remains unknown |

Bounded findings: **high impact** - synchronous GPU readback remains in the application path, so kernel ratios do not establish GUI frame rates. **Medium impact** - quick RCMSE remains a histogram approximation, and GPU similarity still uses 16 bins versus the CPU's 256. **Medium impact** - block sampling can miss small regions; the resulting reference distribution is an estimate. No reduction in similarity bins, RCMSE scales, or compression tier accuracy was introduced by these optimizations.

The quality gates cover non-normative scope, explicit assumptions, changed-path coverage, dimensionally consistent calculations, EOF/concurrency/cache checks, source-and-executable provenance, and these bounded findings. Observed min/max values are timing bounds for these repetitions, not statistical confidence intervals. Complete GUI FPS, cold-disk latency and performance on other native platforms are unknown.

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

- **Malware Analysis**: Identify packed/encrypted sections, detect suspicious entropy patterns
- **Firmware Analysis**: Find compressed regions, locate file systems, identify anomalies
- **Forensics**: Detect hidden data, identify file fragments, find injected content
- **Reverse Engineering**: Understand binary structure, locate interesting regions
- **Data Recovery**: Locate file boundaries in raw disk images
- **Security Research**: Analyze encryption patterns, study packing techniques
- **CTF Competitions**: Quickly identify steganography, hidden data, or unusual structures

## References

- Lyda, R., & Hamrock, J. (2007). "Using Entropy Analysis to Find Encrypted and Packed Malware." *IEEE Security & Privacy*, 5(2), 40-45.
- Costa, M., et al. (2002). "Multiscale entropy analysis of complex physiologic time series." *Physical Review Letters*, 89(6).
- Hilbert, D. (1891). "Über die stetige Abbildung einer Linie auf ein Flächenstück." *Mathematische Annalen*, 38, 459-460.

## License

MIT License - See [LICENSE](LICENSE) for details.

## Acknowledgments

- Inspired by binary visualization research and tools like binvis.io and Veles
- Uses [egui](https://github.com/emilk/egui) for the immediate mode GUI
- GPU compute via [wgpu](https://github.com/gfx-rs/wgpu)
- Portable SIMD via [wide](https://github.com/Lokathor/wide)
- Compression via [xz2](https://github.com/alexcrichton/xz2-rs) (LZMA2) and [zstd](https://github.com/gyscos/zstd-rs)
- File dialogs via [rfd](https://github.com/PolyMeilex/rfd)
- Parallel processing via [rayon](https://github.com/rayon-rs/rayon)
