//! Native GPU benchmark, including upload, dispatch and synchronous readback.
//! cargo run --release --example gpu_performance -- [repetitions]
use apeiron::gpu::{GpuRenderer, GpuVizMode};
use std::{hint::black_box, time::Instant};

fn measure<T>(name: &str, repetitions: usize, mut work: impl FnMut() -> T) {
    black_box(work());
    let mut times: Vec<f64> = (0..repetitions)
        .map(|_| {
            let start = Instant::now();
            black_box(work());
            start.elapsed().as_secs_f64() * 1000.0
        })
        .collect();
    times.sort_by(f64::total_cmp);
    println!(
        "{name},{:.6},{:.6},{:.6}",
        times[times.len() / 2],
        times[0],
        times[times.len() - 1]
    );
}

fn main() {
    let repetitions = std::env::args()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(7)
        .max(1);
    let Some(mut renderer) = GpuRenderer::new() else {
        eprintln!("No GPU adapter available");
        std::process::exit(1);
    };
    let mut seed = 0x4d59_5df4_d0f3_3173u64;
    let data: Vec<u8> = (0..8 * 1024 * 1024)
        .map(|_| {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            seed as u8
        })
        .collect();
    println!("workload,median_ms,min_ms,max_ms");
    measure("upload8MiB", repetitions, || {
        renderer.upload_file(&data, 4096)
    });
    for (name, mode, size, dim) in [
        ("hilbert512", GpuVizMode::Hilbert, 512, 4096.0),
        ("similarity2048", GpuVizMode::SimilarityMatrix, 2048, 4096.0),
        ("digraph2048", GpuVizMode::Digraph, 2048, 256.0),
        ("phase2048", GpuVizMode::BytePhaseSpace, 2048, 256.0),
        ("digraph256", GpuVizMode::Digraph, 256, 256.0),
        ("phase256", GpuVizMode::BytePhaseSpace, 256, 256.0),
    ] {
        measure(name, repetitions, || {
            renderer.render(mode, size, size, 0.0, 0.0, dim, dim)
        });
    }
}
