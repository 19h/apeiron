// Digraph visualization compute shader
// Shows byte pair transition frequencies in a 256x256 grid

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
@group(0) @binding(3) var<storage, read> digraph: array<u32>; // 256x256 counts

// Convert HSV to RGB
fn hsv_to_rgb(h: f32, s: f32, v: f32) -> vec3<f32> {
    let h_mod = (h % 360.0) / 60.0;
    let c = v * s;
    let x = c * (1.0 - abs(h_mod % 2.0 - 1.0));
    let m = v - c;
    
    var rgb: vec3<f32>;
    if h_mod < 1.0 {
        rgb = vec3<f32>(c, x, 0.0);
    } else if h_mod < 2.0 {
        rgb = vec3<f32>(x, c, 0.0);
    } else if h_mod < 3.0 {
        rgb = vec3<f32>(0.0, c, x);
    } else if h_mod < 4.0 {
        rgb = vec3<f32>(0.0, x, c);
    } else if h_mod < 5.0 {
        rgb = vec3<f32>(x, 0.0, c);
    } else {
        rgb = vec3<f32>(c, 0.0, x);
    }
    
    return rgb + vec3<f32>(m, m, m);
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
    
    // Digraph uses 256x256 world space
    let dg_x = i32(world_x);
    let dg_y = i32(world_y);
    
    var color = vec4<f32>(0.031, 0.031, 0.047, 1.0); // Dark background
    
    if dg_x >= 0 && dg_x < 256 && dg_y >= 0 && dg_y < 256 {
        let count = digraph[u32(dg_y * 256 + dg_x)];
        
        if count > 0u {
            // Find max count for normalization (approximate - use file_size as proxy)
            let max_count = max(uniforms.file_size / 256u, 1u);
            let sqrt_max = sqrt(f32(max_count));
            
            // Use sqrt for better contrast distribution
            let intensity = pow(sqrt(f32(count)) / sqrt_max, 0.6);
            
            // Color by byte values with full spectrum
            let hue = f32(dg_x + dg_y) / 512.0 * 300.0;
            let sat = 0.85 - intensity * 0.2;
            let val = 0.15 + intensity * 0.85;
            
            let rgb = hsv_to_rgb(hue, sat, val);
            color = vec4<f32>(rgb, 1.0);
        }
    }
    
    textureStore(output_texture, vec2<i32>(i32(tex_x), i32(tex_y)), color);
}
