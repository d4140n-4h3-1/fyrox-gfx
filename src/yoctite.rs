//! Yoctite in the renderer: the scene's meshes kept in Yoctite's geometry pool, and drawn from it
//! cluster by cluster, at the level of detail each cluster needs from the camera.
//!
//! Every surface is built into clusters at every level of detail once (`yoctite::build`), by its
//! data, so the thousand copies of a tree share one; its vertices and clusters go into the pool
//! (`fyrox_graphics_wgpu::yoctite::GeometryPool`), and each mesh node's surfaces are placed in
//! it as instances, which move with their nodes and go when they do. Each frame, the clusters to
//! draw are chosen - instances out of the view left out, then for each the cut through its
//! clusters whose error looks no bigger than a pixel - on the GPU
//! (`fyrox_graphics_wgpu::yoctite::Culler`), without the CPU seeing the list.
//!
//! [`Mode::Render`] draws them into the G-Buffer, a draw for each material, in place of the
//! renderer's own draw of those meshes (`Renderer::set_taken_over_meshes`): lit, shadowed,
//! fogged and smoothed as everything else is. Only what it can draw as the engine does is taken
//! over: visible meshes, not skinned and without blend shapes, drawn deferred, every surface's
//! material of the engine's standard or glTF shader, without a lightmap. The rest stays the
//! renderer's.
//!
//! [`Mode::View`] instead lays the clusters over the frame, each in a colour of its own: to see
//! them, and which levels of detail are drawn. It can choose them on the CPU (`yoctite::select`)
//! as well, the reference the GPU's choice is held to.

use fyrox::{
    core::{
        algebra::{Matrix4, Point3, Vector2, Vector3},
        color::Color,
        log::Log,
        pool::Handle,
        sstorage::ImmutableString,
    },
    fxhash::{FxHashMap, FxHashSet},
    graph::SceneGraph,
    graphics::{
        error::FrameworkError, gpu_program::SamplerFallback, gpu_program::ShaderPropertyKind,
        gpu_program::ShaderResourceKind, gpu_texture::GpuTexture, sampler::GpuSampler, stats::RenderPassStatistics,
    },
    material::{shader::Shader, Material, MaterialProperty, MaterialResource},
    renderer::{
        cache::shader::{binding, property, PropertyGroup, RenderMaterial, RenderPassContainer},
        make_viewport_matrix,
        observer::{Observer, ObserverPosition},
        resources::RendererResources,
        cache::texture::TextureCache,
        GBufferPassContext, SceneRenderPass, SceneRenderPassContext, TakenOverMeshes,
    },
    asset::manager::ResourceManager,
    scene::{
        collider::BitMask,
        graph::Graph,
        mesh::{
            buffer::{VertexAttributeUsage, VertexReadTrait},
            surface::Surface,
            Mesh, RenderPath,
        },
        node::Node,
        Scene,
    },
};
use fyrox_graphics_wgpu::{
    server::WgpuGraphicsServer,
    yoctite::{
        ClusterView, CullParameters, Culler, GBufferDrawer, GeometryPool, InstanceId, MaterialParams, MeshId,
        PoolVertex, YoctiteMaterial,
    },
};
use std::{
    any::TypeId,
    collections::hash_map::Entry,
    sync::Arc,
    time::{Duration, Instant},
};
use yoctite::ClusterDag;

/// Lays the cluster view over the frame where it drew anything.
const WGSL: &str = include_str!("shaders/yoctite_view_wgsl.shader");

/// How many frames apart the statistics are logged.
const STATS_EVERY: u64 = 300;

/// What Yoctite does in the renderer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Mode {
    /// Draws the meshes it can into the G-Buffer, in place of the renderer.
    Render,
    /// Lays the clusters over the frame in colours of their own, chosen on the GPU or, `on_gpu`
    /// false, on the CPU.
    View {
        /// Whether the clusters are chosen on the GPU.
        on_gpu: bool,
    },
}

/// A mesh node's surface placed in the pool.
struct Placed {
    instance: InstanceId,
    /// Its surface's data, to tell when it is given other data.
    data: u64,
    /// Its material, by key, and the slot it is drawn with.
    material: u64,
    /// Where it was placed, to tell when it moves.
    transform: Matrix4<f32>,
    /// The last frame it was seen in, to take out what has gone.
    seen: u64,
}

/// How a material is drawn into the G-Buffer: as the engine's glTF shader does, or its standard
/// shader.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Shading {
    Gltf,
    Standard,
}

/// A material in use, and the slot its instances are numbered with.
struct Slot {
    slot: u32,
    material: MaterialResource,
    shading: Shading,
    /// How many placed surfaces are drawn with it; the slot is given back at none.
    users: u32,
}

/// The materials drawn with, by key, and their slots.
#[derive(Default)]
struct Materials {
    by_key: FxHashMap<u64, Slot>,
    free: Vec<u32>,
    next: u32,
}

impl Materials {
    /// The slot of `material`, given one if it has none, counted as used once more.
    fn take(&mut self, material: &MaterialResource, shading: Shading) -> u32 {
        let Self { by_key, free, next } = self;
        let slot = by_key.entry(material.key()).or_insert_with(|| Slot {
            slot: free.pop().unwrap_or_else(|| {
                *next += 1;
                *next - 1
            }),
            material: material.clone(),
            shading,
            users: 0,
        });
        slot.users += 1;
        slot.slot
    }

    /// Counts `key`'s material as used once less, giving its slot back at none.
    fn give_back(&mut self, key: u64) {
        if let Some(slot) = self.by_key.get_mut(&key) {
            slot.users -= 1;
            if slot.users == 0 {
                self.free.push(slot.slot);
                self.by_key.remove(&key);
            }
        }
    }

    #[cfg(test)]
    fn slot(&self, key: u64) -> u32 {
        self.by_key.get(&key).map_or(0, |slot| slot.slot)
    }

    /// How many slots there are, in use or not.
    fn slots(&self) -> u32 {
        self.next
    }
}

/// How long a scene's pool is kept while the scene is not drawn: a scene switched off for longer,
/// or gone, lets its GPU memory go, and is placed again if it is drawn again.
const KEEP_UNDRAWN_SCENE: Duration = Duration::from_secs(10);

/// How many times a scene's pool is brought in step with it while a mesh in it has no surface
/// placed, before the mesh is taken out: not at once, so that one coming and going - a level of
/// detail shown and hidden - is not uploaded again each time.
const KEEP_UNUSED_MESH: u64 = 600;

/// How often, in passes drawn, built clusters no scene's pool holds are let go.
const FORGET_DAGS_EVERY: u64 = 300;

/// Yoctite in the renderer: see the module's description. [`crate::GraphicsEffects::with_yoctite`]
/// and [`crate::GraphicsEffects::with_yoctite_view`] add it.
pub struct YoctitePass {
    source_type_id: TypeId,
    mode: Mode,
    /// The meshes it draws, scene by scene, which the renderer leaves out of its own draw.
    taken_over: Option<TakenOverMeshes>,
    drawer: Option<GBufferDrawer>,
    view: Option<ClusterView>,
    shader: Option<RenderPassContainer>,
    /// Each surface's clusters as built, by its data: built once, whichever scenes it is in.
    dags: FxHashMap<u64, Arc<ClusterDag>>,
    /// What is kept for each scene drawn: node handles are only unique within their scene, and
    /// the GPU draws everything in a pool, so each scene has a pool of its own.
    scenes: FxHashMap<Handle<Scene>, ScenePool>,
    /// Passes drawn, to let go of built clusters now and then.
    passes: u64,
    failed: bool,
}

impl std::fmt::Debug for YoctitePass {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("YoctitePass")
            .field("mode", &self.mode)
            .field("scenes", &self.scenes.len())
            .field("built", &self.dags.len())
            .field("failed", &self.failed)
            .finish()
    }
}

/// A scene's meshes in Yoctite: its own pool, what is placed in it, the materials they are drawn
/// with and the culler that chooses their clusters.
struct ScenePool {
    pool: GeometryPool,
    culler: Option<Culler>,
    /// Each surface's mesh in this pool, by the surface's data, and the last time a surface of it
    /// was placed (see `frame`).
    meshes: FxHashMap<u64, (MeshId, u64)>,
    placed: FxHashMap<(Handle<Node>, usize), Placed>,
    materials: Materials,
    /// How many times it has been brought in step with its scene.
    frame: u64,
    /// When it was last drawn, to let it go when its scene no longer is.
    drawn: Instant,
    overflow_reported: bool,
}

impl ScenePool {
    fn new(server: &WgpuGraphicsServer) -> Self {
        Self {
            pool: GeometryPool::new(&server.state.device, 1 << 16),
            culler: None,
            meshes: Default::default(),
            placed: Default::default(),
            materials: Default::default(),
            frame: 0,
            drawn: Instant::now(),
            overflow_reported: false,
        }
    }
}

impl YoctitePass {
    /// The pass, in `mode`; drawing in place of the renderer, it tells the renderer which meshes
    /// it draws through `taken_over`.
    pub(crate) fn new(source_type_id: TypeId, mode: Mode, taken_over: Option<TakenOverMeshes>) -> Self {
        Self {
            source_type_id,
            mode,
            taken_over,
            drawer: None,
            view: None,
            shader: None,
            dags: Default::default(),
            scenes: Default::default(),
            passes: 0,
            failed: false,
        }
    }

    /// The pool of `scene`, made if it has none; and the pools of scenes not drawn for a while let
    /// go, with the meshes they had taken over.
    fn scene_pool(&mut self, server: &WgpuGraphicsServer, scene: Handle<Scene>) -> &mut ScenePool {
        let now = Instant::now();
        let taken_over = &self.taken_over;
        self.scenes.retain(|&handle, pool| {
            let keep = handle == scene || now.duration_since(pool.drawn) < KEEP_UNDRAWN_SCENE;
            if !keep {
                if let Some(taken_over) = taken_over {
                    taken_over.borrow_mut().remove(&handle);
                }
            }
            keep
        });
        // Clusters built for surfaces no pool holds any more: their meshes are gone from the scenes.
        self.passes += 1;
        if self.passes.is_multiple_of(FORGET_DAGS_EVERY) {
            let scenes = &self.scenes;
            self.dags
                .retain(|data, _| scenes.values().any(|pool| pool.meshes.contains_key(data)));
        }
        let pool = self.scenes.entry(scene).or_insert_with(|| ScenePool::new(server));
        pool.drawn = now;
        pool
    }

    /// Gives every scene's meshes back to the renderer.
    fn give_back_all(&self) {
        if let Some(taken_over) = &self.taken_over {
            taken_over.borrow_mut().clear();
        }
    }
}

/// A surface's vertices as the pool keeps them, and its triangles' indices.
fn read_surface(surface: &Surface) -> (Vec<PoolVertex>, Vec<u32>) {
    let data = surface.data();
    let data = data.data_ref();
    let vertices = data
        .vertex_buffer
        .iter()
        .map(|v| PoolVertex {
            position: v.read_3_f32(VertexAttributeUsage::Position).unwrap_or_default().into(),
            normal: v.read_3_f32(VertexAttributeUsage::Normal).unwrap_or_default().into(),
            uv: v.read_2_f32(VertexAttributeUsage::TexCoord0).unwrap_or_default().into(),
            tangent: v.read_4_f32(VertexAttributeUsage::Tangent).unwrap_or_default().into(),
        })
        .collect::<Vec<_>>();
    let count = vertices.len() as u32;
    let indices = data
        .geometry_buffer
        .iter()
        .filter(|t| t.0.iter().all(|&i| i < count))
        .flat_map(|t| t.0)
        .collect();
    (vertices, indices)
}

/// `m`'s top three rows, row by row, as an instance is placed with.
fn rows(m: &Matrix4<f32>) -> [f32; 12] {
    let mut out = [0.0; 12];
    for row in 0..3 {
        for column in 0..4 {
            out[row * 4 + column] = m[(row, column)];
        }
    }
    out
}

/// Whether a sphere is at least partly inside the frustum of `view_projection` (OpenGL clip
/// space), by its six planes.
fn in_view(view_projection: &Matrix4<f32>, center: Vector3<f32>, radius: f32) -> bool {
    let row = |i: usize| view_projection.row(i).transpose();
    let (r0, r1, r2, r3) = (row(0), row(1), row(2), row(3));
    [r3 + r0, r3 - r0, r3 + r1, r3 - r1, r3 + r2, r3 - r2].iter().all(|plane| {
        let normal = Vector3::new(plane.x, plane.y, plane.z);
        let length = normal.norm();
        length == 0.0 || (normal.dot(&center) + plane.w) / length >= -radius
    })
}

/// How `material` is drawn into the G-Buffer, if Yoctite can draw it as the engine does: a
/// loaded material of the engine's glTF or standard shader, without a lightmap.
fn shading(material: &MaterialResource) -> Option<Shading> {
    let state = material.state();
    let material = state.data_ref()?;
    let shader = material.shader().state();
    let shader = shader.data_ref()?;
    let shading = match shader.definition.name.as_str() {
        "GLTFShader" => Shading::Gltf,
        "StandardShader" => Shading::Standard,
        _ => return None,
    };
    let lightmapped = material
        .texture_ref("lightmapTexture")
        .is_some_and(|binding| binding.value.is_some());
    (!lightmapped).then_some(shading)
}

/// `material`'s value of the property `name`: its own, or else its shader's default.
fn value_of(material: &Material, name: &str) -> Option<MaterialProperty> {
    if let Some(own) = material
        .property_group_ref(ImmutableString::new("properties"))
        .and_then(|group| group.property_ref(name))
    {
        return Some(own.clone());
    }
    let shader = material.shader().state();
    let shader = shader.data_ref()?;
    shader.definition.resources.iter().find_map(|resource| match &resource.kind {
        ShaderResourceKind::PropertyGroup(properties) => properties.iter().find(|p| p.name.as_str() == name).and_then(
            |p| match &p.kind {
                ShaderPropertyKind::Float { value } => Some(MaterialProperty::Float(*value)),
                ShaderPropertyKind::UInt { value } => Some(MaterialProperty::UInt(*value)),
                ShaderPropertyKind::Vector2 { value } => Some(MaterialProperty::Vector2(*value)),
                ShaderPropertyKind::Vector3 { value } => Some(MaterialProperty::Vector3(*value)),
                ShaderPropertyKind::Color { r, g, b, a } => Some(MaterialProperty::Color(Color::from_rgba(*r, *g, *b, *a))),
                _ => None,
            },
        ),
        _ => None,
    })
}

/// The GPU texture `material` binds as `name`, with its sampler: its own, once loaded, or else
/// the stub its shader falls back to, as the renderer does.
fn texture_of(
    material: &Material,
    name: &str,
    texture_cache: &mut TextureCache,
    server: &dyn fyrox::graphics::server::GraphicsServer,
    resource_manager: &ResourceManager,
    resources: &RendererResources,
) -> (GpuTexture, GpuSampler) {
    if let Some(texture) = material.texture_ref(name).and_then(|binding| binding.value.clone()) {
        if let Some(data) = texture_cache.get(server, resource_manager, &texture) {
            return (data.gpu_texture.clone(), data.gpu_sampler.clone());
        }
    }
    let fallback = material
        .shader()
        .state()
        .data_ref()
        .and_then(|shader| {
            shader.definition.resources.iter().find_map(|resource| match &resource.kind {
                ShaderResourceKind::Texture { fallback, .. } if resource.name.as_str() == name => Some(*fallback),
                _ => None,
            })
        })
        .unwrap_or(SamplerFallback::White);
    let stub = match fallback {
        SamplerFallback::Black => &resources.black_dummy,
        SamplerFallback::Normal => &resources.normal_dummy,
        SamplerFallback::White | SamplerFallback::Volume => &resources.white_dummy,
    };
    (stub.clone(), resources.linear_wrap_sampler.clone())
}

/// A material in a slot, as Yoctite's G-Buffer draw reads it.
fn resolve(slot: &Slot, ctx: &mut GBufferPassContext) -> Option<YoctiteMaterial> {
    let state = slot.material.state();
    let material = state.data_ref()?;
    let float = |name, default: f32| match value_of(material, name) {
        Some(MaterialProperty::Float(v)) => v,
        _ => default,
    };
    let diffuse_color = match value_of(material, "diffuseColor") {
        Some(MaterialProperty::Color(c)) => c.as_frgba().into(),
        _ => [1.0; 4],
    };
    let emission_strength = match value_of(material, "emissionStrength") {
        Some(MaterialProperty::Vector3(v)) => v.into(),
        _ => [0.0; 3],
    };
    let tex_coord_scale = match value_of(material, "texCoordScale") {
        Some(MaterialProperty::Vector2(v)) => v.into(),
        _ => [1.0; 2],
    };
    let layer_index = match value_of(material, "layerIndex") {
        Some(MaterialProperty::UInt(v)) => v,
        _ => 0,
    };
    // The glTF shader reads metallic and roughness from the blue and green of one texture,
    // times factors; the standard shader from the red of one each.
    let (metallic, roughness, metallic_channel, roughness_channel, metallic_factor, roughness_factor) = match slot.shading {
        Shading::Gltf => (
            "metallicRoughnessTexture",
            "metallicRoughnessTexture",
            2,
            1,
            float("metallicFactor", 1.0),
            float("roughnessFactor", 1.0),
        ),
        Shading::Standard => ("metallicTexture", "roughnessTexture", 0, 0, 1.0, 1.0),
    };
    let mut texture = |name: &str| {
        texture_of(material, name, ctx.texture_cache, ctx.server, ctx.resource_manager, ctx.renderer_resources)
    };
    let textures = [
        texture("diffuseTexture"),
        texture("normalTexture"),
        texture(metallic),
        texture(roughness),
        texture("aoTexture"),
        texture("emissionTexture"),
    ];
    Some(YoctiteMaterial {
        slot: slot.slot,
        params: MaterialParams {
            diffuse_color,
            emission_strength,
            metallic_factor,
            tex_coord_scale,
            roughness_factor,
            layer_index,
            metallic_channel,
            roughness_channel,
            ..Default::default()
        },
        textures,
    })
}

/// An error as big as a pixel, as the observer sees: the view's height in radians over its height
/// in pixels.
fn pixel_threshold(observer: &Observer) -> f32 {
    let fov = 2.0 * (1.0 / observer.position.projection_matrix[(1, 1)]).atan();
    fov / observer.viewport.h().max(1) as f32
}

/// The observer's view and projection, column by column.
fn view_projection_of(observer: &Observer) -> [[f32; 4]; 4] {
    let mut out = [[0.0; 4]; 4];
    for (column, values) in observer.position.view_projection_matrix.column_iter().zip(out.iter_mut()) {
        values.copy_from_slice(column.as_slice());
    }
    out
}

/// The mesh nodes the renderer would draw from `position` with `render_mask`, as far as nodes
/// go: in the render mask, visible, enabled, and not left out by a level of detail - nor below one that is, since
/// the renderer skips that whole branch. Yoctite draws no more than these, so that a mesh it takes
/// over is one the renderer would have drawn.
fn drawable_meshes<'a>(
    graph: &'a Graph,
    position: &ObserverPosition,
    render_mask: BitMask,
) -> Vec<(Handle<Node>, &'a Node, &'a Mesh)> {
    // Levels of detail as the renderer chooses them: by distance from the observer, between its
    // near and far planes; where an object is in several levels, the last one decides.
    let mut lod_visible: FxHashMap<u32, bool> = FxHashMap::default();
    for (_, node) in graph.pair_iter() {
        let Some(group) = node.lod_group() else {
            continue;
        };
        for level in group.levels.iter() {
            for &object in level.objects.iter() {
                if let Ok(object_ref) = graph.try_get_node(object) {
                    let distance = position.translation.metric_distance(&object_ref.global_position());
                    let normalized = (distance - position.z_near) / (position.z_far - position.z_near);
                    lod_visible.insert(object.index(), normalized >= level.begin() && normalized <= level.end());
                }
            }
        }
    }
    let mut meshes = Vec::new();
    let mut stack = vec![graph.root()];
    while let Some(handle) = stack.pop() {
        if lod_visible.get(&handle.index()) == Some(&false) {
            continue;
        }
        let Ok(node) = graph.try_get_node(handle) else {
            continue;
        };
        stack.extend_from_slice(node.children());
        if let Some(mesh) = node.cast::<Mesh>() {
            if *node.render_mask & render_mask != BitMask::none()
                && node.global_visibility()
                && node.is_globally_enabled()
            {
                meshes.push((handle, node, mesh));
            }
        }
    }
    meshes
}

/// Keeps `scene`'s pool in step with the scene's meshes as `observer` would draw them: new
/// surfaces built (once, into `dags`) and uploaded, new mesh nodes placed, moved ones moved, gone
/// ones taken out. Drawing in place of the renderer (`rendering`), only meshes whose materials it
/// can draw are placed, and each with its material's slot. Returns the mesh nodes placed.
fn sync(
    scene: &mut ScenePool,
    dags: &mut FxHashMap<u64, Arc<ClusterDag>>,
    rendering: bool,
    server: &WgpuGraphicsServer,
    graph: &Graph,
    observer: &Observer,
) -> FxHashSet<Handle<Node>> {
    let device = &server.state.device;
    let queue = &server.state.queue;
    let ScenePool {
        pool,
        meshes,
        placed: placements,
        materials,
        frame,
        ..
    } = scene;
    *frame += 1;
    let frame = *frame;
    let mut newly_built = (0, 0usize, 0.0f64);
    let mut nodes = FxHashSet::default();
    for (handle, node, mesh) in drawable_meshes(graph, &observer.position, observer.render_mask) {
        // Skinned meshes, those with blend shapes and those drawn forward stay the renderer's.
        if mesh.render_path() != RenderPath::Deferred
            || mesh
                .surfaces()
                .iter()
                .any(|s| !s.bones().is_empty() || s.data().data_ref().blend_shapes_container.is_some())
        {
            continue;
        }
        let shadings: Option<Vec<Shading>> = if rendering {
            mesh.surfaces().iter().map(|s| shading(s.material())).collect()
        } else {
            Some(vec![Shading::Standard; mesh.surfaces().len()])
        };
        let Some(shadings) = shadings else {
            continue;
        };
        nodes.insert(handle);
        let transform = node.global_transform();
        for ((n, surface), shading) in mesh.surfaces().iter().enumerate().zip(shadings) {
            let data = surface.data().key();
            let mesh = match meshes.entry(data) {
                Entry::Occupied(mut mesh) => {
                    mesh.get_mut().1 = frame;
                    mesh.get().0
                }
                Entry::Vacant(place) => {
                    let (vertices, indices) = read_surface(surface);
                    let dag = match dags.entry(data) {
                        Entry::Occupied(dag) => dag.get().clone(),
                        Entry::Vacant(place) => {
                            let positions: Vec<[f32; 3]> = vertices.iter().map(|v| v.position).collect();
                            // With its texture coordinates, so a level of detail that drags its
                            // texture across it counts the drag as error.
                            let uvs: Vec<[f32; 2]> = vertices.iter().map(|v| v.uv).collect();
                            let started = Instant::now();
                            let dag = yoctite::build(
                                &yoctite::Mesh::new(&positions, &indices).with_texture_coordinates(&uvs),
                                Default::default(),
                            );
                            newly_built.0 += 1;
                            newly_built.1 += indices.len() / 3;
                            newly_built.2 += started.elapsed().as_secs_f64();
                            place.insert(Arc::new(dag)).clone()
                        }
                    };
                    place.insert((pool.add_mesh(device, queue, &dag, &vertices), frame)).0
                }
            };
            let material = if rendering { surface.material().key() } else { 0 };
            match placements.get_mut(&(handle, n)) {
                Some(placed) if placed.data == data && placed.material == material => {
                    if placed.transform != transform {
                        pool.move_instance(queue, placed.instance, rows(&transform));
                        placed.transform = transform;
                    }
                    placed.seen = frame;
                }
                other => {
                    if let Some(old) = other {
                        pool.remove_instance(queue, old.instance);
                        if rendering {
                            materials.give_back(old.material);
                        }
                    }
                    let slot = if rendering { materials.take(surface.material(), shading) } else { 0 };
                    let instance = pool.add_instance(device, queue, mesh, rows(&transform), slot);
                    placements.insert(
                        (handle, n),
                        Placed {
                            instance,
                            data,
                            material,
                            transform,
                            seen: frame,
                        },
                    );
                }
            }
        }
    }
    placements.retain(|_, placed| {
        let keep = placed.seen == frame;
        if !keep {
            pool.remove_instance(queue, placed.instance);
            if rendering {
                materials.give_back(placed.material);
            }
        }
        keep
    });
    // Meshes no surface has been placed with for a while: none of their instances is left.
    meshes.retain(|_, &mut (mesh, used)| {
        let keep = frame - used < KEEP_UNUSED_MESH;
        if !keep {
            pool.remove_mesh(queue, mesh);
        }
        keep
    });
    if newly_built.0 > 0 {
        let stats = pool.stats();
        Log::info(format!(
            "Yoctite: built {} surfaces ({} triangles) in {:.2} s; the scene's pool holds {} meshes, {} instances, \
             {} clusters, {:.1} MB",
            newly_built.0,
            newly_built.1,
            newly_built.2,
            stats.meshes,
            stats.instances,
            stats.clusters,
            stats.bytes as f64 / 1048576.0
        ));
    }
    nodes
}

impl ScenePool {
    /// What the GPU chose, read back a frame or two late, as clusters and triangles; warning once
    /// if its list ran out of room.
    fn chosen(&mut self, what: &str) -> Option<(usize, usize)> {
        let culler = self.culler.as_ref()?;
        culler.stats().map(|stats| {
            if stats.overflowed && !self.overflow_reported {
                Log::warn(format!(
                    "Yoctite {what}: {} clusters chosen, more than the draw list held; it holds {} from now on",
                    stats.clusters,
                    culler.capacity()
                ));
                self.overflow_reported = true;
            }
            (stats.clusters as usize, stats.triangles as usize)
        })
    }
}

/// The clusters to draw from where the camera is, chosen on the CPU, as instance slots and pool
/// clusters, and how many triangles they come to: the reference for the GPU's choice.
fn choose(
    dags: &FxHashMap<u64, Arc<ClusterDag>>,
    scene: &ScenePool,
    view_projection: &Matrix4<f32>,
    eye: Vector3<f32>,
    threshold: f32,
) -> (Vec<[u32; 2]>, usize) {
    let mut list = Vec::new();
    let mut triangles = 0;
    for placed in scene.placed.values() {
        let (Some(dag), Some(&(mesh, _))) = (dags.get(&placed.data), scene.meshes.get(&placed.data)) else {
            continue;
        };
        let Some(mesh) = scene.pool.mesh(mesh) else {
            continue;
        };
        let t = &placed.transform;
        let scale = (0..3).map(|c| t.column(c).xyz().norm()).fold(0.0, f32::max);
        let center = t.transform_point(&Point3::new(mesh.bounds[0], mesh.bounds[1], mesh.bounds[2]));
        if !in_view(view_projection, center.coords, mesh.bounds[3] * scale) {
            continue;
        }
        // In the mesh's own space, where its errors and bounds are.
        let Some(inverse) = t.try_inverse() else {
            continue;
        };
        let local = inverse.transform_point(&Point3::from(eye));
        for c in yoctite::select(dag, [local.x, local.y, local.z], threshold) {
            triangles += dag.clusters[c as usize].triangles.len();
            list.push([placed.instance.0, mesh.first_cluster + c]);
        }
    }
    (list, triangles)
}

impl SceneRenderPass for YoctitePass {
    fn on_gbuffer_render(&mut self, mut ctx: GBufferPassContext) -> Result<RenderPassStatistics, FrameworkError> {
        if self.failed || self.mode != Mode::Render {
            return Ok(Default::default());
        }
        let Some(server) = ctx.server.as_any().downcast_ref::<WgpuGraphicsServer>() else {
            return Ok(Default::default());
        };
        let device = &server.state.device;
        let started = Instant::now();
        let scene_handle = ctx.scene_handle;
        self.scene_pool(server, scene_handle);
        let Self {
            scenes,
            dags,
            taken_over,
            drawer,
            ..
        } = self;
        let scene = scenes.get_mut(&scene_handle).expect("made above");
        let nodes = sync(scene, dags, true, server, &ctx.scene.graph, ctx.observer);
        // From the next frame on the renderer leaves these out of this scene: this one, it drew
        // them already, which drawing them again changes nothing of.
        if let Some(taken_over) = taken_over {
            taken_over.borrow_mut().insert(scene_handle, nodes);
        }
        let materials: Vec<YoctiteMaterial> = scene
            .materials
            .by_key
            .values()
            .filter_map(|slot| resolve(slot, &mut ctx))
            .collect();
        let synced = started.elapsed().as_secs_f64();

        let eye = ctx.observer.position.translation;
        let parameters = CullParameters {
            view_projection: view_projection_of(ctx.observer),
            eye: eye.into(),
            threshold: pixel_threshold(ctx.observer),
            material_slots: scene.materials.slots().max(1),
        };
        let culler = scene.culler.get_or_insert_with(|| Culler::new(device));
        let drawer = drawer.get_or_insert_with(|| GBufferDrawer::new(device));
        let started = Instant::now();
        let draws = match drawer.draw(server, &scene.pool, culler, &parameters, ctx.framebuffer, &materials) {
            Ok(draws) => draws,
            Err(err) => {
                Log::err(format!("Yoctite failed, leaving the meshes to the renderer: {err:?}"));
                self.failed = true;
                self.give_back_all();
                return Ok(Default::default());
            }
        };
        let drew = started.elapsed().as_secs_f64();
        let (clusters, triangles) = scene.chosen("").unwrap_or_default();
        if scene.frame % STATS_EVERY == 1 {
            Log::info(format!(
                "Yoctite ({scene_handle}): {clusters} clusters, {triangles} triangles in {} draws ({} materials); \
                 the CPU took {:.2} ms keeping up with the scene and its materials and {:.2} ms choosing and drawing",
                draws.draw_calls,
                materials.len(),
                synced * 1000.0,
                drew * 1000.0
            ));
        }
        let mut stats = RenderPassStatistics::default();
        stats.draw_calls += draws.draw_calls as usize;
        stats.triangles_rendered += triangles;
        Ok(stats)
    }

    fn on_hdr_render(&mut self, ctx: SceneRenderPassContext) -> Result<RenderPassStatistics, FrameworkError> {
        let Mode::View { on_gpu } = self.mode else {
            return Ok(Default::default());
        };
        if self.failed {
            return Ok(Default::default());
        }
        let Some(server) = ctx.server.as_any().downcast_ref::<WgpuGraphicsServer>() else {
            return Ok(Default::default());
        };
        if self.shader.is_none() {
            let shader = Shader::from_string(WGSL)
                .map_err(|e| FrameworkError::Custom(format!("Yoctite view shader: {e:?}")))?;
            self.shader = Some(RenderPassContainer::new(ctx.server, &shader)?);
        }
        let device = &server.state.device;
        let started = Instant::now();
        let scene_handle = ctx.scene_handle;
        self.scene_pool(server, scene_handle);
        let Self { scenes, dags, view, .. } = self;
        let scene = scenes.get_mut(&scene_handle).expect("made above");
        sync(scene, dags, false, server, &ctx.scene.graph, ctx.observer);
        let synced = started.elapsed().as_secs_f64();

        let position = &ctx.observer.position;
        let viewport = ctx.observer.viewport;
        let threshold = pixel_threshold(ctx.observer);
        let view_projection = view_projection_of(ctx.observer);
        let eye = position.translation;
        let (width, height) = (viewport.w() as u32, viewport.h() as u32);
        let view = view.get_or_insert_with(|| ClusterView::new(device));
        let started = Instant::now();
        let (drawn, cpu_chosen) = if on_gpu {
            let culler = scene.culler.get_or_insert_with(|| Culler::new(device));
            let drawn = view.draw_culled(server, &scene.pool, culler, view_projection, eye.into(), threshold, width, height);
            (drawn, None)
        } else {
            let (list, triangles) = choose(dags, scene, &position.view_projection_matrix, eye, threshold);
            let drawn = view.draw(server, &scene.pool, view_projection, width, height, &list);
            (drawn, Some((list.len(), triangles)))
        };
        let chose = started.elapsed().as_secs_f64();
        let drawn = match drawn {
            Ok(texture) => texture,
            Err(err) => {
                Log::err(format!("Yoctite view failed: {err:?}"));
                self.failed = true;
                return Ok(Default::default());
            }
        };
        let (clusters, triangles) = cpu_chosen.or_else(|| scene.chosen("view")).unwrap_or_default();
        if scene.frame % STATS_EVERY == 1 {
            Log::info(format!(
                "Yoctite view ({scene_handle}): {} clusters, {} triangles, chosen on the {}; the pool kept up with \
                 the scene in {:.2} ms, choosing and drawing took the CPU {:.2} ms",
                clusters,
                triangles,
                if on_gpu { "GPU" } else { "CPU" },
                synced * 1000.0,
                chose * 1000.0
            ));
        }

        let size = Vector2::new(viewport.w() as f32, viewport.h() as f32);
        let world_view_projection = make_viewport_matrix(viewport);
        let properties = PropertyGroup::from([
            property("worldViewProjection", &world_view_projection),
            property("screenSize", &size),
        ]);
        let material = RenderMaterial::from([
            binding("clusters", (&drawn, &ctx.renderer_resources.nearest_clamp_sampler)),
            binding("properties", &properties),
        ]);
        let shader = self.shader.as_ref().expect("made above");
        let composited = shader.run_pass(
            1,
            &ImmutableString::new("Primary"),
            ctx.framebuffer,
            &ctx.renderer_resources.quad,
            viewport,
            &material,
            ctx.uniform_buffer_cache,
            Default::default(),
            None,
        )?;
        let mut stats = RenderPassStatistics::default();
        stats += composited;
        // And the clusters' own draw.
        stats.draw_calls += 1;
        stats.triangles_rendered += triangles;
        Ok(stats)
    }

    fn source_type_id(&self) -> TypeId {
        self.source_type_id
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use fyrox::core::algebra::Perspective3;

    #[test]
    fn spheres_in_and_out_of_view() {
        // Looking down -z, as OpenGL's cameras do.
        let projection = Perspective3::new(1.0, 1.0, 0.1, 100.0).to_homogeneous();
        assert!(in_view(&projection, Vector3::new(0.0, 0.0, -10.0), 1.0));
        assert!(!in_view(&projection, Vector3::new(0.0, 0.0, 10.0), 1.0), "behind");
        assert!(!in_view(&projection, Vector3::new(50.0, 0.0, -10.0), 1.0), "off to the side");
        assert!(in_view(&projection, Vector3::new(11.0, 0.0, -10.0), 6.0), "partly in");
        assert!(!in_view(&projection, Vector3::new(0.0, 0.0, -200.0), 1.0), "past the far plane");
    }

    #[test]
    fn rows_are_the_top_of_the_matrix() {
        let m = Matrix4::new_translation(&Vector3::new(1.0, 2.0, 3.0));
        assert_eq!(rows(&m), [1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 2.0, 0.0, 0.0, 1.0, 3.0]);
    }

    #[test]
    fn drawn_meshes_are_those_the_renderer_would_draw() {
        use fyrox::scene::{
            base::{BaseBuilder, LevelOfDetail, LodGroup},
            mesh::MeshBuilder,
            pivot::PivotBuilder,
            transform::TransformBuilder,
        };
        let mut graph = Graph::new();
        let at = |z: f32| TransformBuilder::new().with_local_position(Vector3::new(0.0, 0.0, z)).build();
        let plain = MeshBuilder::new(BaseBuilder::new()).build(&mut graph).to_base::<Node>();
        let masked = MeshBuilder::new(BaseBuilder::new()).build(&mut graph).to_base::<Node>();
        *graph[masked].render_mask = BitMask(0b10);
        let hidden = MeshBuilder::new(BaseBuilder::new().with_visibility(false)).build(&mut graph).to_base::<Node>();
        let disabled = MeshBuilder::new(BaseBuilder::new().with_enabled(false)).build(&mut graph).to_base::<Node>();
        let near = MeshBuilder::new(BaseBuilder::new().with_local_transform(at(1.0))).build(&mut graph).to_base::<Node>();
        let under_far = MeshBuilder::new(BaseBuilder::new()).build(&mut graph).to_base::<Node>();
        let far = MeshBuilder::new(BaseBuilder::new().with_local_transform(at(1.0)).with_child(under_far))
            .build(&mut graph).to_base::<Node>();
        PivotBuilder::new(BaseBuilder::new().with_lod_group(LodGroup {
            levels: vec![
                LevelOfDetail::new(0.0, 0.1, vec![near]),
                LevelOfDetail::new(0.1, 1.0, vec![far]),
            ],
        }))
        .build(&mut graph);
        graph.update_hierarchical_data();

        let position = ObserverPosition {
            z_near: 0.1,
            z_far: 100.0,
            ..Default::default()
        };
        let drawn = |mask| -> FxHashSet<Handle<Node>> {
            drawable_meshes(&graph, &position, mask).into_iter().map(|(h, ..)| h).collect()
        };
        let all = drawn(BitMask(0b01));
        assert!(all.contains(&plain));
        assert!(!all.contains(&masked), "outside the camera's render mask");
        assert!(!all.contains(&hidden), "hidden");
        assert!(!all.contains(&disabled), "disabled");
        assert!(all.contains(&near), "its level of detail is the one shown up close");
        assert!(!all.contains(&far), "its level of detail is not shown up close");
        assert!(!all.contains(&under_far), "under a level of detail not shown");
        assert!(drawn(BitMask(0b11)).contains(&masked), "inside the render mask");
    }

    #[test]
    fn material_slots_are_given_back_and_reused() {
        let a = MaterialResource::new_embedded(Material::standard());
        let b = MaterialResource::new_embedded(Material::standard());
        let mut materials = Materials::default();
        assert_eq!(materials.take(&a, Shading::Standard), 0);
        assert_eq!(materials.take(&a, Shading::Standard), 0, "the same material, the same slot");
        assert_eq!(materials.take(&b, Shading::Standard), 1);
        materials.give_back(a.key());
        assert_eq!(materials.slot(a.key()), 0, "still used once");
        materials.give_back(a.key());
        assert!(!materials.by_key.contains_key(&a.key()), "no longer used");
        let c = MaterialResource::new_embedded(Material::standard());
        assert_eq!(materials.take(&c, Shading::Standard), 0, "its slot used again");
        assert_eq!(materials.slots(), 2);
    }
}
