// Similarity Matrix visualization compute shader
// Shows recurrence plot based on chi-squared distance between byte histograms

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

// Compute histogram for a window - returns counts for 16 buckets (simplified)
fn compute_histogram_16(start: u32) -> array<u32, 16> {
    var hist: array<u32, 16>;
    for (var i = 0u; i < 16u; i = i + 1u) {
        hist[i] = 0u;
    }
    
    let end = min(start + WINDOW_SIZE, uniforms.file_size);
    for (var i = start; i < end; i = i + 1u) {
        let byte = get_byte(i);
        let bucket = byte / 16u; // 256 bytes -> 16 buckets
        hist[bucket] = hist[bucket] + 1u;
    }
    
    return hist;
}

// Compute chi-squared distance between two histograms
fn chi_squared_distance(hist_x: array<u32, 16>, hist_y: array<u32, 16>) -> f32 {
    var total_x = 0u;
    var total_y = 0u;
    
    for (var i = 0u; i < 16u; i = i + 1u) {
        total_x = total_x + hist_x[i];
        total_y = total_y + hist_y[i];
    }
    
    if total_x == 0u || total_y == 0u {
        return 0.0;
    }
    
    var chi_sq = 0.0;
    let total_x_f = f32(total_x);
    let total_y_f = f32(total_y);
    
    for (var i = 0u; i < 16u; i = i + 1u) {
        let px = f32(hist_x[i]) / total_x_f;
        let py = f32(hist_y[i]) / total_y_f;
        let sum = px + py;
        if sum > 0.0 {
            chi_sq = chi_sq + (px - py) * (px - py) / sum;
        }
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
    
    // Map texture coordinates to world coordinates
    let scale_x = (uniforms.world_max_x - uniforms.world_min_x) / f32(uniforms.tex_width);
    let scale_y = (uniforms.world_max_y - uniforms.world_min_y) / f32(uniforms.tex_height);
    let world_x = uniforms.world_min_x + f32(tex_x) * scale_x;
    let world_y = uniforms.world_min_y + f32(tex_y) * scale_y;
    
    let dim = f32(uniforms.dimension);
    
    var color = vec4<f32>(0.039, 0.039, 0.059, 1.0); // Dark background
    
    if world_x >= 0.0 && world_y >= 0.0 && world_x < dim && world_y < dim {
        // Map world coordinates to file positions
        let pos_x = u32((world_x / dim) * f32(uniforms.file_size));
        let pos_y = u32((world_y / dim) * f32(uniforms.file_size));
        
        let hist_x = compute_histogram_16(pos_x);
        let hist_y = compute_histogram_16(pos_y);
        
        let chi_sq = chi_squared_distance(hist_x, hist_y);
        
        // Convert distance to similarity (chi_sq ranges from 0 to 2)
        let similarity = 1.0 - sqrt(chi_sq / 2.0);
        
        let is_diagonal = abs(i32(pos_x) - i32(pos_y)) < i32(WINDOW_SIZE);
        let rgb = similarity_to_color(similarity, is_diagonal);
        color = vec4<f32>(rgb, 1.0);
    }
    
    textureStore(output_texture, vec2<i32>(i32(tex_x), i32(tex_y)), color);
}
