// Hilbert curve visualization compute shader
// Maps file bytes to a Hilbert curve with forensic color scheme

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

const ANALYSIS_WINDOW: u32 = 64u;

// Rotate/flip quadrant for Hilbert curve
fn rot(n: u32, x: ptr<function, u32>, y: ptr<function, u32>, rx: u32, ry: u32) {
    if ry == 0u {
        if rx == 1u {
            *x = n - 1u - *x;
            *y = n - 1u - *y;
        }
        let t = *x;
        *x = *y;
        *y = t;
    }
}

// Convert (x, y) to Hilbert curve index
fn xy2d(n: u32, x_in: u32, y_in: u32) -> u32 {
    var x = x_in;
    var y = y_in;
    var d = 0u;
    var s = n / 2u;
    
    while s > 0u {
        let rx = select(0u, 1u, (x & s) > 0u);
        let ry = select(0u, 1u, (y & s) > 0u);
        d = d + s * s * ((3u * rx) ^ ry);
        rot(s, &x, &y, rx, ry);
        s = s / 2u;
    }
    
    return d;
}

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

// Analyze bytes for forensic color mapping
struct ByteAnalysis {
    text_ratio: f32,
    high_ratio: f32,
    null_ratio: f32,
    variation: f32,
}

fn analyze_bytes(start: u32) -> ByteAnalysis {
    var text_chars = 0u;
    var high_bits = 0u;
    var nulls = 0u;
    var variation = 0u;
    var prev_byte = 0u;
    var first = true;
    
    let end = min(start + ANALYSIS_WINDOW, uniforms.file_size);
    let count = end - start;
    
    if count == 0u {
        return ByteAnalysis(0.0, 0.0, 0.0, 0.0);
    }
    
    for (var i = start; i < end; i = i + 1u) {
        let byte = get_byte(i);
        
        // Count printable ASCII (32-126)
        if byte >= 32u && byte <= 126u {
            text_chars = text_chars + 1u;
        }
        // Count high-bit bytes
        if byte > 127u {
            high_bits = high_bits + 1u;
        }
        // Count nulls
        if byte == 0u {
            nulls = nulls + 1u;
        }
        // Compute variation
        if !first {
            let diff = select(byte - prev_byte, prev_byte - byte, prev_byte > byte);
            variation = variation + diff;
        }
        prev_byte = byte;
        first = false;
    }
    
    let count_f = f32(count);
    return ByteAnalysis(
        f32(text_chars) / count_f,
        f32(high_bits) / count_f,
        f32(nulls) / count_f,
        (f32(variation) / count_f) / 128.0
    );
}

// Map analysis to RGB color using forensic color scheme
fn analysis_to_color(a: ByteAnalysis) -> vec3<f32> {
    // Padding / Zeroes -> Deep Blue
    if a.null_ratio > 0.9 {
        let intensity = min(0.2 + 0.3 * a.null_ratio, 1.0);
        return vec3<f32>(0.0, 0.0, intensity);
    }
    
    // High Entropy / Encryption -> Red/Orange
    if a.variation > 0.5 && a.high_ratio > 0.25 {
        return vec3<f32>(1.0, 0.2, 0.0);
    }
    
    // ASCII Text -> Cyan
    if a.text_ratio > 0.85 {
        let intensity = min(0.8 * a.variation + 0.2, 1.0);
        return vec3<f32>(0.0, intensity, intensity);
    }
    
    // Mixed/Moderate text -> Yellow/Green gradient
    if a.text_ratio > 0.5 {
        let t = (a.text_ratio - 0.5) / 0.35;
        let intensity = min(0.5 + 0.5 * a.variation, 1.0);
        return vec3<f32>(intensity * (1.0 - t), intensity, intensity * t);
    }
    
    // Code / Machine Instructions -> Green
    let intensity = min(0.5 + 0.5 * a.variation, 1.0);
    return vec3<f32>(0.0, intensity, 0.0);
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
    
    var color = vec4<f32>(0.05, 0.05, 0.05, 1.0); // Dark background
    
    let x = u32(world_x);
    let y = u32(world_y);
    let n = uniforms.dimension;
    
    if x < n && y < n {
        let d = xy2d(n, x, y);
        
        if d < uniforms.file_size {
            let analysis = analyze_bytes(d);
            let rgb = analysis_to_color(analysis);
            color = vec4<f32>(rgb, 1.0);
        }
    }
    
    textureStore(output_texture, vec2<i32>(i32(tex_x), i32(tex_y)), color);
}
