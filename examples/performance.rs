//! Reproducible, dependency-free release benchmark. Run with:
//! cargo run --release --example performance -- [repetitions] [filter]
use std::hint::black_box;
use std::time::Instant;

use apeiron::{analysis, hilbert, viz};
use egui::Vec2;

fn measure<T>(name: &str, repetitions: usize, filter: &str, mut work: impl FnMut() -> T) {
    if !name.contains(filter) {
        return;
    }
    black_box(work());
    let mut times = Vec::with_capacity(repetitions);
    for _ in 0..repetitions {
        let start = Instant::now();
        black_box(work());
        times.push(start.elapsed().as_secs_f64() * 1_000.0);
    }
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
    let filter = std::env::args().nth(2).unwrap_or_default();
    let mut seed = 0x4d59_5df4_d0f3_3173u64;
    let random: Vec<u8> = (0..8 * 1024 * 1024)
        .map(|_| {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            seed as u8
        })
        .collect();
    let mixed: Vec<u8> = random
        .iter()
        .enumerate()
        .map(|(i, &b)| match (i / 4096) % 4 {
            0 => 0,
            1 => b'A' + b % 26,
            2 => (i % 256) as u8,
            _ => b,
        })
        .collect();
    let offsets: Vec<usize> = (0..16_384).map(|i| i * 64).collect();
    let reference = analysis::byte_distribution(&mixed);
    // Initialize the global pool before timed work.
    rayon::ThreadPoolBuilder::new().build_global().unwrap();
    hilbert::curve::warmup_luts();
    eprintln!(
        "threads={}, repetitions={repetitions}, data_bytes={}",
        rayon::current_num_threads(),
        random.len()
    );
    println!("workload,median_ms,min_ms,max_ms");
    for (label, data) in [("random", &random), ("mixed", &mixed)] {
        measure(&format!("entropy64_{label}"), repetitions, &filter, || {
            offsets
                .iter()
                .map(|&i| analysis::calculate_entropy(black_box(&data[i..i + 64])))
                .sum::<f64>()
        });
        measure(
            &format!("kolmogorov128_{label}"),
            repetitions,
            &filter,
            || {
                offsets[..2048]
                    .iter()
                    .map(|&i| {
                        analysis::calculate_kolmogorov_complexity(black_box(&data[i..i + 128]))
                    })
                    .sum::<f64>()
            },
        );
        measure(&format!("rcmse256_{label}"), repetitions, &filter, || {
            offsets[..256]
                .iter()
                .map(|&i| analysis::calculate_rcmse_quick(black_box(&data[i..i + 256])))
                .sum::<f32>()
        });
    }
    measure("jsd512", repetitions, &filter, || {
        viz::generate_jsd_pixels(
            &mixed,
            &reference,
            4096,
            mixed.len() as u64,
            512,
            Vec2::ZERO,
            8.0,
            8.0,
            false,
        )
    });
    measure("jsd512_zoom", repetitions, &filter, || {
        viz::generate_jsd_pixels(
            &mixed,
            &reference,
            4096,
            mixed.len() as u64,
            512,
            Vec2::new(10.0, 10.0),
            0.25,
            0.25,
            false,
        )
    });
    measure("hilbert512", repetitions, &filter, || {
        viz::generate_hilbert_pixels(&mixed, 4096, mixed.len() as u64, 512, Vec2::ZERO, 8.0, 8.0)
    });
    measure("linear512", repetitions, &filter, || {
        viz::generate_linear_entropy_pixels(
            &mixed,
            4096,
            mixed.len() as u64,
            512,
            Vec2::ZERO,
            8.0,
            8.0,
        )
    });
    for (label, data) in [("random", &random), ("mixed", &mixed)] {
        measure(
            &format!("similarity512_{label}"),
            repetitions,
            &filter,
            || {
                viz::generate_similarity_matrix_pixels(
                    data,
                    data.len() as u64,
                    512,
                    Vec2::ZERO,
                    8.0,
                    8.0,
                    4096,
                )
            },
        );
        measure(
            &format!("similarity2048_{label}"),
            repetitions,
            &filter,
            || {
                viz::generate_similarity_matrix_pixels(
                    data,
                    data.len() as u64,
                    2048,
                    Vec2::ZERO,
                    2.0,
                    2.0,
                    4096,
                )
            },
        );
    }
    measure("digraph2048", repetitions, &filter, || {
        viz::generate_digraph_pixels(&mixed, mixed.len() as u64, 2048, Vec2::ZERO, 0.125, 0.125)
    });
    measure("phase2048", repetitions, &filter, || {
        viz::generate_byte_phase_space_pixels(
            &mixed,
            mixed.len() as u64,
            2048,
            Vec2::ZERO,
            0.125,
            0.125,
        )
    });
}
