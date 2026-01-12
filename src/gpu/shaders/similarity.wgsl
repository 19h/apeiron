// Similarity Matrix visualization compute shader
// Shows recurrence plot based on chi-squared distance between byte histograms
//
// Optimizations:
// - Loop unrolling for histogram computation
// - Precomputed reciprocals for normalization
// - Reduced floating point operations in inner loop

struct Uniforms {
    tex_width: u32,
    tex_height: u32,
    file_size: u32,
    dimension: u32,
    world_min_x: f32,
    world_min_y: f32,
    world_max_x: f32,
    world_max_y: f32,
}

@group(0) @binding(0) var<uniform> uniforms: Uniforms;
@group(0) @binding(1) var<storage, read> file_data: array<u32>;
@group(0) @binding(2) var output_texture: texture_storage_2d<rgba8unorm, write>;
@group(0) @binding(3) var<storage, read> precomputed: array<u32>;

const WINDOW_SIZE: u32 = 32u;
const COLOR_BACKGROUND: vec4<f32> = vec4<f32>(0.039, 0.039, 0.059, 1.0);

// Get byte at offset (file_data is packed as u32)
fn get_byte(offset: u32) -> u32 {
    if offset >= uniforms.file_size {
        return 0u;
    }
    let word_idx = offset / 4u;
    let byte_idx = offset % 4u;
    let word = file_data[word_idx];
    return (word >> (byte_idx * 8u)) & 0xFFu;
}

// Compute histogram for a window - writes counts for 16 buckets into output pointer
// Using pointer to avoid FXC array-by-value issues on Windows
// Optimized with loop unrolling
fn compute_histogram_16(start: u32, hist: ptr<function, array<u32, 16>>) {
    // Initialize to zero - unrolled for better vectorization
    (*hist)[0] = 0u; (*hist)[1] = 0u; (*hist)[2] = 0u; (*hist)[3] = 0u;
    (*hist)[4] = 0u; (*hist)[5] = 0u; (*hist)[6] = 0u; (*hist)[7] = 0u;
    (*hist)[8] = 0u; (*hist)[9] = 0u; (*hist)[10] = 0u; (*hist)[11] = 0u;
    (*hist)[12] = 0u; (*hist)[13] = 0u; (*hist)[14] = 0u; (*hist)[15] = 0u;
    
    let end = min(start + WINDOW_SIZE, uniforms.file_size);
    var i = start;
    
    // Process 4 bytes at a time
    while i + 4u <= end {
        let b0 = get_byte(i) >> 4u;      // /16 via shift
        let b1 = get_byte(i + 1u) >> 4u;
        let b2 = get_byte(i + 2u) >> 4u;
        let b3 = get_byte(i + 3u) >> 4u;
        (*hist)[b0] = (*hist)[b0] + 1u;
        (*hist)[b1] = (*hist)[b1] + 1u;
        (*hist)[b2] = (*hist)[b2] + 1u;
        (*hist)[b3] = (*hist)[b3] + 1u;
        i = i + 4u;
    }
    
    // Handle remainder
    while i < end {
        let bucket = get_byte(i) >> 4u;
        (*hist)[bucket] = (*hist)[bucket] + 1u;
        i = i + 1u;
    }
}

// Compute chi-squared distance between two histograms
// Using pointers to avoid FXC array-by-value issues on Windows
// Optimized with unrolled loops and precomputed reciprocals
fn chi_squared_distance(hist_x: ptr<function, array<u32, 16>>, hist_y: ptr<function, array<u32, 16>>) -> f32 {
    // Compute totals with unrolled loop
    var total_x = (*hist_x)[0] + (*hist_x)[1] + (*hist_x)[2] + (*hist_x)[3];
    total_x += (*hist_x)[4] + (*hist_x)[5] + (*hist_x)[6] + (*hist_x)[7];
    total_x += (*hist_x)[8] + (*hist_x)[9] + (*hist_x)[10] + (*hist_x)[11];
    total_x += (*hist_x)[12] + (*hist_x)[13] + (*hist_x)[14] + (*hist_x)[15];
    
    var total_y = (*hist_y)[0] + (*hist_y)[1] + (*hist_y)[2] + (*hist_y)[3];
    total_y += (*hist_y)[4] + (*hist_y)[5] + (*hist_y)[6] + (*hist_y)[7];
    total_y += (*hist_y)[8] + (*hist_y)[9] + (*hist_y)[10] + (*hist_y)[11];
    total_y += (*hist_y)[12] + (*hist_y)[13] + (*hist_y)[14] + (*hist_y)[15];
    
    if total_x == 0u || total_y == 0u {
        return 0.0;
    }
    
    // Precompute reciprocals
    let inv_total_x = 1.0 / f32(total_x);
    let inv_total_y = 1.0 / f32(total_y);
    
    var chi_sq = 0.0;
    
    // Unrolled chi-squared computation
    for (var i = 0u; i < 16u; i = i + 4u) {
        let px0 = f32((*hist_x)[i]) * inv_total_x;
        let py0 = f32((*hist_y)[i]) * inv_total_y;
        let sum0 = px0 + py0;
        let diff0 = px0 - py0;
        chi_sq += select(0.0, diff0 * diff0 / sum0, sum0 > 0.0);
        
        let px1 = f32((*hist_x)[i + 1u]) * inv_total_x;
        let py1 = f32((*hist_y)[i + 1u]) * inv_total_y;
        let sum1 = px1 + py1;
        let diff1 = px1 - py1;
        chi_sq += select(0.0, diff1 * diff1 / sum1, sum1 > 0.0);
        
        let px2 = f32((*hist_x)[i + 2u]) * inv_total_x;
        let py2 = f32((*hist_y)[i + 2u]) * inv_total_y;
        let sum2 = px2 + py2;
        let diff2 = px2 - py2;
        chi_sq += select(0.0, diff2 * diff2 / sum2, sum2 > 0.0);
        
        let px3 = f32((*hist_x)[i + 3u]) * inv_total_x;
        let py3 = f32((*hist_y)[i + 3u]) * inv_total_y;
        let sum3 = px3 + py3;
        let diff3 = px3 - py3;
        chi_sq += select(0.0, diff3 * diff3 / sum3, sum3 > 0.0);
    }
    
    return chi_sq;
}

// Convert similarity to color using inferno-style colormap
fn similarity_to_color(similarity: f32, is_diagonal: bool) -> vec3<f32> {
    // Apply gamma curve to enhance contrast
    let t = pow(clamp(similarity, 0.0, 1.0), 0.4);
    
    if is_diagonal {
        // Diagonal elements - bright cyan/white
        return vec3<f32>(0.7 + t * 0.3, 0.86 + t * 0.14, 1.0);
    }
    
    // Inferno-inspired colormap
    if t < 0.2 {
        let s = t / 0.2;
        return vec3<f32>(s * 0.25, 0.0, s * 0.35);
    } else if t < 0.4 {
        let s = (t - 0.2) / 0.2;
        return vec3<f32>(0.25 + s * 0.45, 0.0, 0.35 + s * 0.1);
    } else if t < 0.6 {
        let s = (t - 0.4) / 0.2;
        return vec3<f32>(0.7 + s * 0.25, s * 0.25, 0.45 - s * 0.35);
    } else if t < 0.8 {
        let s = (t - 0.6) / 0.2;
        return vec3<f32>(0.95 + s * 0.05, 0.25 + s * 0.45, 0.1 + s * 0.1);
    } else {
        let s = (t - 0.8) / 0.2;
        return vec3<f32>(1.0, 0.7 + s * 0.3, 0.2 + s * 0.7);
    }
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let tex_x = global_id.x;
    let tex_y = global_id.y;
    
    if tex_x >= uniforms.tex_width || tex_y >= uniforms.tex_height {
        return;
    }
    
    // Precompute scale factors
    let inv_tex_width = 1.0 / f32(uniforms.tex_width);
    let inv_tex_height = 1.0 / f32(uniforms.tex_height);
    let scale_x = (uniforms.world_max_x - uniforms.world_min_x) * inv_tex_width;
    let scale_y = (uniforms.world_max_y - uniforms.world_min_y) * inv_tex_height;
    let world_x = fma(f32(tex_x), scale_x, uniforms.world_min_x);
    let world_y = fma(f32(tex_y), scale_y, uniforms.world_min_y);
    
    let dim = f32(uniforms.dimension);
    
    // Early exit for out-of-bounds
    if world_x < 0.0 || world_y < 0.0 || world_x >= dim || world_y >= dim {
        textureStore(output_texture, vec2<i32>(i32(tex_x), i32(tex_y)), COLOR_BACKGROUND);
        return;
    }
    
    // Map world coordinates to file positions
    let inv_dim = 1.0 / dim;
    let file_scale = f32(uniforms.file_size) * inv_dim;
    let pos_x = u32(world_x * file_scale);
    let pos_y = u32(world_y * file_scale);
    
    // Compute histograms using pointers (FXC-compatible)
    var hist_x: array<u32, 16>;
    var hist_y: array<u32, 16>;
    compute_histogram_16(pos_x, &hist_x);
    compute_histogram_16(pos_y, &hist_y);
    
    let chi_sq = chi_squared_distance(&hist_x, &hist_y);
    
    // Convert distance to similarity (chi_sq ranges from 0 to 2)
    // sqrt(chi_sq/2) = sqrt(chi_sq) * 0.7071...
    let similarity = 1.0 - sqrt(chi_sq) * 0.70710678;
    
    let is_diagonal = abs(i32(pos_x) - i32(pos_y)) < i32(WINDOW_SIZE);
    let rgb = similarity_to_color(similarity, is_diagonal);
    textureStore(output_texture, vec2<i32>(i32(tex_x), i32(tex_y)), vec4<f32>(rgb, 1.0));
}
