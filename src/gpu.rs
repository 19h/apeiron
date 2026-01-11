//! GPU-accelerated visualization renderer using wgpu compute shaders.

use bytemuck::{Pod, Zeroable};

use wgpu::util::DeviceExt;

/// Visualization mode for GPU rendering.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GpuVizMode {
    Hilbert,
    Digraph,
    BytePhaseSpace,
    SimilarityMatrix,
}

/// Uniform parameters passed to compute shaders.
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
struct Uniforms {
    /// Output texture width.
    tex_width: u32,
    /// Output texture height.
    tex_height: u32,
    /// File size in bytes.
    file_size: u32,
    /// Hilbert curve dimension (power of 2).
    dimension: u32,
    /// Viewport world min X.
    world_min_x: f32,
    /// Viewport world min Y.
    world_min_y: f32,
    /// Viewport world max X.
    world_max_x: f32,
    /// Viewport world max Y.
    world_max_y: f32,
}

/// GPU renderer for entropy visualizations.
pub struct GpuRenderer {
    device: wgpu::Device,
    queue: wgpu::Queue,
    /// Pipeline for Hilbert visualization.
    hilbert_pipeline: wgpu::ComputePipeline,
    /// Pipeline for Digraph visualization.
    digraph_pipeline: wgpu::ComputePipeline,
    /// Pipeline for BytePhaseSpace visualization.
    phase_space_pipeline: wgpu::ComputePipeline,
    /// Pipeline for SimilarityMatrix visualization.
    similarity_pipeline: wgpu::ComputePipeline,
    /// Bind group layout shared by all pipelines.
    bind_group_layout: wgpu::BindGroupLayout,
    /// Currently loaded file data buffer.
    file_buffer: Option<wgpu::Buffer>,
    /// Current file size.
    file_size: u32,
    /// Current Hilbert dimension.
    dimension: u32,
    /// Precomputed digraph buffer (256x256 u32 counts).
    digraph_buffer: Option<wgpu::Buffer>,
    /// Precomputed phase space buffer (256x256 x2 u32: last_pos, count).
    phase_space_buffer: Option<wgpu::Buffer>,
}

impl GpuRenderer {
    /// Create a new GPU renderer.
    pub fn new() -> Option<Self> {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            ..Default::default()
        });

        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: None,
            force_fallback_adapter: false,
        }))?;

        let (device, queue) = pollster::block_on(adapter.request_device(
            &wgpu::DeviceDescriptor {
                label: Some("Apeiron GPU"),
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::default(),
                memory_hints: wgpu::MemoryHints::Performance,
            },
            None,
        ))
        .ok()?;

        // Create bind group layout shared by all compute shaders
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Compute Bind Group Layout"),
            entries: &[
                // Uniform buffer
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // File data buffer (read-only)
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Output texture (write)
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::StorageTexture {
                        access: wgpu::StorageTextureAccess::WriteOnly,
                        format: wgpu::TextureFormat::Rgba8Unorm,
                        view_dimension: wgpu::TextureViewDimension::D2,
                    },
                    count: None,
                },
                // Precomputed data buffer (for digraph/phase space)
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Compute Pipeline Layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        // Create compute pipelines for each visualization
        let hilbert_pipeline = Self::create_pipeline(
            &device,
            &pipeline_layout,
            include_str!("shaders/hilbert.wgsl"),
        );
        let digraph_pipeline = Self::create_pipeline(
            &device,
            &pipeline_layout,
            include_str!("shaders/digraph.wgsl"),
        );
        let phase_space_pipeline = Self::create_pipeline(
            &device,
            &pipeline_layout,
            include_str!("shaders/phase_space.wgsl"),
        );
        let similarity_pipeline = Self::create_pipeline(
            &device,
            &pipeline_layout,
            include_str!("shaders/similarity.wgsl"),
        );

        Some(Self {
            device,
            queue,
            hilbert_pipeline,
            digraph_pipeline,
            phase_space_pipeline,
            similarity_pipeline,
            bind_group_layout,
            file_buffer: None,
            file_size: 0,
            dimension: 0,
            digraph_buffer: None,
            phase_space_buffer: None,
        })
    }

    fn create_pipeline(
        device: &wgpu::Device,
        layout: &wgpu::PipelineLayout,
        shader_source: &str,
    ) -> wgpu::ComputePipeline {
        let shader_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Compute Shader"),
            source: wgpu::ShaderSource::Wgsl(shader_source.into()),
        });

        device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Compute Pipeline"),
            layout: Some(layout),
            module: &shader_module,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        })
    }

    /// Maximum file size for GPU upload (128MB is wgpu's max_storage_buffer_binding_size)
    const MAX_GPU_FILE_SIZE: usize = 120 * 1024 * 1024;

    /// Upload file data to GPU and precompute derived data.
    /// Returns false if file is too large for GPU.
    pub fn upload_file(&mut self, data: &[u8], dimension: u64) -> bool {
        self.file_size = data.len() as u32;
        self.dimension = dimension as u32;

        // Check if file is too large for GPU buffer
        if data.len() > Self::MAX_GPU_FILE_SIZE {
            println!(
                "File too large for GPU ({} MB > {} MB limit), using CPU fallback",
                data.len() / (1024 * 1024),
                Self::MAX_GPU_FILE_SIZE / (1024 * 1024)
            );
            self.file_buffer = None;
            self.digraph_buffer = None;
            self.phase_space_buffer = None;
            return false;
        }

        // Upload raw file data
        self.file_buffer = Some(self.device.create_buffer_init(
            &wgpu::util::BufferInitDescriptor {
                label: Some("File Data Buffer"),
                contents: data,
                usage: wgpu::BufferUsages::STORAGE,
            },
        ));

        // Precompute digraph (256x256 transition counts)
        let mut digraph = vec![0u32; 256 * 256];
        if data.len() >= 2 {
            for window in data.windows(2) {
                let from = window[0] as usize;
                let to = window[1] as usize;
                digraph[from * 256 + to] += 1;
            }
        }
        self.digraph_buffer = Some(self.device.create_buffer_init(
            &wgpu::util::BufferInitDescriptor {
                label: Some("Digraph Buffer"),
                contents: bytemuck::cast_slice(&digraph),
                usage: wgpu::BufferUsages::STORAGE,
            },
        ));

        // Precompute phase space (256x256 x2: last_position, count)
        let mut phase_space = vec![0u32; 256 * 256 * 2];
        if data.len() >= 2 {
            for (i, window) in data.windows(2).enumerate() {
                let x = window[0] as usize;
                let y = window[1] as usize;
                let idx = (y * 256 + x) * 2;
                phase_space[idx] = i as u32; // last_position
                phase_space[idx + 1] += 1; // count
            }
        }
        self.phase_space_buffer = Some(self.device.create_buffer_init(
            &wgpu::util::BufferInitDescriptor {
                label: Some("Phase Space Buffer"),
                contents: bytemuck::cast_slice(&phase_space),
                usage: wgpu::BufferUsages::STORAGE,
            },
        ));

        true
    }

    /// Render visualization to RGBA pixel data.
    pub fn render(
        &self,
        mode: GpuVizMode,
        tex_width: u32,
        tex_height: u32,
        world_min_x: f32,
        world_min_y: f32,
        world_max_x: f32,
        world_max_y: f32,
    ) -> Vec<u8> {
        let Some(file_buffer) = &self.file_buffer else {
            return vec![0u8; (tex_width * tex_height * 4) as usize];
        };

        // Create uniform buffer
        let uniforms = Uniforms {
            tex_width,
            tex_height,
            file_size: self.file_size,
            dimension: self.dimension,
            world_min_x,
            world_min_y,
            world_max_x,
            world_max_y,
        };

        let uniform_buffer = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Uniform Buffer"),
                contents: bytemuck::bytes_of(&uniforms),
                usage: wgpu::BufferUsages::UNIFORM,
            });

        // Create output texture
        let output_texture = self.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Output Texture"),
            size: wgpu::Extent3d {
                width: tex_width,
                height: tex_height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let output_view = output_texture.create_view(&Default::default());

        // Create output buffer for reading back
        // bytes_per_row must be aligned to COPY_BYTES_PER_ROW_ALIGNMENT (256)
        let unpadded_bytes_per_row = tex_width * 4;
        let align = wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;
        let padded_bytes_per_row = (unpadded_bytes_per_row + align - 1) / align * align;
        let output_buffer_size = (padded_bytes_per_row * tex_height) as u64;
        let output_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Output Buffer"),
            size: output_buffer_size,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        // Select pipeline and precomputed buffer based on mode
        let (pipeline, precomputed_buffer) = match mode {
            GpuVizMode::Hilbert => (&self.hilbert_pipeline, file_buffer),
            GpuVizMode::Digraph => (
                &self.digraph_pipeline,
                self.digraph_buffer.as_ref().unwrap_or(file_buffer),
            ),
            GpuVizMode::BytePhaseSpace => (
                &self.phase_space_pipeline,
                self.phase_space_buffer.as_ref().unwrap_or(file_buffer),
            ),
            GpuVizMode::SimilarityMatrix => (&self.similarity_pipeline, file_buffer),
        };

        // Create bind group
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Compute Bind Group"),
            layout: &self.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: uniform_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: file_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(&output_view),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: precomputed_buffer.as_entire_binding(),
                },
            ],
        });

        // Encode and submit compute pass
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Compute Encoder"),
            });

        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Compute Pass"),
                timestamp_writes: None,
            });
            compute_pass.set_pipeline(pipeline);
            compute_pass.set_bind_group(0, &bind_group, &[]);
            // Dispatch workgroups (8x8 threads per group)
            let workgroups_x = (tex_width + 7) / 8;
            let workgroups_y = (tex_height + 7) / 8;
            compute_pass.dispatch_workgroups(workgroups_x, workgroups_y, 1);
        }

        // Copy texture to buffer
        encoder.copy_texture_to_buffer(
            wgpu::ImageCopyTexture {
                texture: &output_texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::ImageCopyBuffer {
                buffer: &output_buffer,
                layout: wgpu::ImageDataLayout {
                    offset: 0,
                    bytes_per_row: Some(padded_bytes_per_row),
                    rows_per_image: Some(tex_height),
                },
            },
            wgpu::Extent3d {
                width: tex_width,
                height: tex_height,
                depth_or_array_layers: 1,
            },
        );

        self.queue.submit(std::iter::once(encoder.finish()));

        // Read back the result
        let buffer_slice = output_buffer.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
            tx.send(result).unwrap();
        });
        self.device.poll(wgpu::Maintain::Wait);
        rx.recv().unwrap().unwrap();

        let data = buffer_slice.get_mapped_range();

        // Remove row padding if present
        let result = if padded_bytes_per_row != unpadded_bytes_per_row {
            let mut unpacked = Vec::with_capacity((tex_width * tex_height * 4) as usize);
            for row in 0..tex_height {
                let start = (row * padded_bytes_per_row) as usize;
                let end = start + unpadded_bytes_per_row as usize;
                unpacked.extend_from_slice(&data[start..end]);
            }
            unpacked
        } else {
            data.to_vec()
        };

        drop(data);
        output_buffer.unmap();

        result
    }

    /// Check if GPU renderer is available and has file loaded.
    pub fn is_ready(&self) -> bool {
        self.file_buffer.is_some()
    }
}
