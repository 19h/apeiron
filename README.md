# Apeiron

Binary file entropy and complexity visualizer using Hilbert curves and advanced analysis techniques.

Apeiron provides visual analysis of binary files through multiple visualization modes, helping identify patterns, encrypted regions, compressed data, and structural anomalies in executables, firmware, and other binary formats.

## Features

- **Multiple Visualization Modes**: Five distinct visualization techniques for comprehensive binary analysis
- **GPU Acceleration**: Hardware-accelerated rendering via wgpu compute shaders
- **Real-time Inspection**: Hover over any region to see detailed byte analysis
- **Hilbert Curve Mapping**: Space-filling curve preserves locality - nearby bytes appear as nearby pixels
- **Pan & Zoom**: Explore large files with smooth navigation
- **Cross-Platform**: Runs on macOS, Linux, and Windows

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

Useful for identifying repeated code blocks, copy-paste patterns, and structural repetition.

### Byte Digraph (DIG)
A 256x256 heatmap showing byte transition frequencies. X-axis is the source byte value, Y-axis is the following byte value:
- **Bright regions**: Frequently occurring byte pairs
- **Dark regions**: Rare or absent transitions
- **Clusters**: Reveal character set usage (ASCII, Unicode, binary patterns)

Effective for distinguishing file types and identifying encoding schemes.

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

Unlike Shannon entropy, Kolmogorov complexity captures algorithmic patterns and is sensitive to compressibility rather than just byte distribution.

## Installation

### Pre-built Binaries
Download the latest release for your platform from the [Releases](../../releases) page:
- `apeiron-macos-arm64` - macOS Apple Silicon
- `apeiron-macos-x86_64` - macOS Intel
- `apeiron-linux-x86_64` - Linux x86_64
- `apeiron-windows-x86_64.exe` - Windows x86_64

### Building from Source

Requirements:
- Rust 1.70+ (install via [rustup](https://rustup.rs))
- On Linux: `libxcb`, `libxkbcommon`, `libgtk-3`

```bash
# Clone the repository
git clone https://github.com/user/apeiron.git
cd apeiron

# Build release version
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
```

## Usage

1. **Open a file**: Drag and drop any binary file onto the window, or click "Open File..."
2. **Navigate**: 
   - Scroll to zoom in/out
   - Click and drag to pan
   - Hover over regions to inspect bytes
3. **Switch modes**: Use the Mode dropdown in the toolbar
4. **Reset view**: Click "Reset View" to fit the visualization to the window

## Controls

| Action | Control |
|--------|---------|
| Zoom | Scroll wheel |
| Pan | Click and drag |
| Inspect | Hover over pixels |
| Open file | Drag & drop or "Open File..." button |
| Reset view | "Reset View" button |
| Help | "Help" button |

## Data Inspector

The right panel shows detailed information about the currently hovered byte position:

- **File Info**: Detected file type and size
- **Cursor Location**: Byte offset in hex and decimal
- **Entropy Analysis**: Shannon entropy (0-8 bits) with visual bar
- **Kolmogorov Complexity**: Compression ratio percentage with interpretation
- **Hex Preview**: 64-byte hex dump with ASCII representation
- **String Preview**: Extracted ASCII strings (if present)

## Technical Details

### Entropy Calculation
Shannon entropy is calculated over a 64-byte sliding window:
```
H = -Σ p(x) * log2(p(x))
```
where p(x) is the probability of byte value x in the window.

### Kolmogorov Complexity Approximation
Complexity is approximated using DEFLATE compression:
```
K(x) ≈ len(compress(x)) / len(x)
```
Pre-computed at file load time for performance (sampled every 64 bytes with 128-byte windows).

### Hilbert Curve
The Hilbert curve dimension is chosen as the smallest power of 2 where n² >= file_size. This ensures all bytes can be mapped while maintaining the locality-preserving property.

### GPU Acceleration
When available, visualization rendering uses wgpu compute shaders for parallel pixel generation. Falls back to CPU (with rayon parallelization) when GPU is unavailable or for modes not implemented on GPU.

## File Type Detection

Apeiron automatically detects common file types via magic bytes:
- Executables: PE (EXE/DLL), ELF, Mach-O
- Archives: ZIP, RAR, GZIP, BZIP2, 7-Zip
- Images: PNG, JPEG, GIF
- Documents: PDF
- Media: MP4/MOV, WAV/AVI (RIFF)
- Databases: SQLite

## Use Cases

- **Malware Analysis**: Identify packed/encrypted sections, locate code caves
- **Firmware Analysis**: Find compressed regions, locate file systems
- **Forensics**: Detect hidden data, identify file fragments
- **Reverse Engineering**: Understand binary structure, find interesting regions
- **Data Recovery**: Locate file boundaries in raw disk images
- **Security Research**: Analyze encryption patterns, find vulnerabilities

## Performance

- Large files (100MB+) are handled efficiently via viewport-aware rendering
- Kolmogorov complexity is pre-computed at load time using parallel processing
- GPU acceleration significantly speeds up Hilbert, Digraph, Phase Space, and Similarity Matrix modes
- Texture regeneration is throttled to prevent excessive recomputation during navigation

## License

MIT License - See [LICENSE](LICENSE) for details.

## Acknowledgments

- Inspired by binary visualization research and tools like binvis.io
- Uses [egui](https://github.com/emilk/egui) for the immediate mode GUI
- GPU compute via [wgpu](https://github.com/gfx-rs/wgpu)
