//! Yoctite's cluster view: the scene's meshes kept in Yoctite's geometry pool, and drawn from it
//! cluster by cluster, each cluster in a colour of its own, at the level of detail it would be
//! drawn at from the camera.
//!
//! Every surface is built into clusters at every level of detail once (`yoctite::build`), by its
//! data, so the thousand copies of a tree share one; its vertices and clusters go into the pool
//! (`fyrox_graphics_wgpu::yoctite::GeometryPool`), and each mesh node's surfaces are placed in
//! it as instances, which move with their nodes and go when they do. Each frame, the clusters to
//! draw are chosen - instances out of the view left out, then for each the cut through its
//! clusters whose error looks no bigger than a pixel - and drawn in one draw. They are chosen on
//! the GPU (`fyrox_graphics_wgpu::yoctite::Culler`), the draw made from its list without the CPU
//! seeing it; or, as a reference, on the CPU (`yoctite::select`).

use fyrox::{
    core::{
        algebra::{Matrix4, Point3, Vector2, Vector3},
        log::Log,
        pool::Handle,
        sstorage::ImmutableString,
    },
    fxhash::FxHashMap,
    graph::SceneGraph,
    graphics::{error::FrameworkError, stats::RenderPassStatistics},
    material::shader::Shader,
    renderer::{
        cache::shader::{binding, property, PropertyGroup, RenderMaterial, RenderPassContainer},
        make_viewport_matrix, SceneRenderPass, SceneRenderPassContext,
    },
    scene::{
        mesh::{
            buffer::{VertexAttributeUsage, VertexReadTrait},
            surface::Surface,
            Mesh,
        },
        node::Node,
    },
};
use fyrox_graphics_wgpu::{
    server::WgpuGraphicsServer,
    yoctite::{ClusterView, Culler, GeometryPool, InstanceId, MeshId, PoolVertex},
};
use std::{any::TypeId, collections::hash_map::Entry, sync::Arc, time::Instant};
use yoctite::ClusterDag;

/// Lays the cluster view over the frame where it drew anything.
const WGSL: &str = include_str!("shaders/yoctite_view_wgsl.shader");

/// How many frames apart the view's statistics are logged.
const STATS_EVERY: u64 = 300;

/// A surface's clusters, in the pool and as built.
struct Built {
    mesh: MeshId,
    dag: Arc<ClusterDag>,
}

/// A mesh node's surface placed in the pool.
struct Placed {
    instance: InstanceId,
    /// Its surface's data, to tell when it is given other data.
    data: u64,
    /// Where it was placed, to tell when it moves.
    transform: Matrix4<f32>,
    /// The last frame it was seen in, to take out what has gone.
    seen: u64,
}

/// The cluster view: see the module's description. [`crate::GraphicsEffects::with_yoctite_view`]
/// adds it.
pub struct YoctiteViewPass {
    source_type_id: TypeId,
    pool: Option<GeometryPool>,
    view: Option<ClusterView>,
    shader: Option<RenderPassContainer>,
    built: FxHashMap<u64, Built>,
    placed: FxHashMap<(Handle<Node>, usize), Placed>,
    frame: u64,
    failed: bool,
    /// Whether the clusters are chosen on the GPU, or on the CPU, as a reference.
    on_gpu: bool,
    culler: Option<Culler>,
    overflow_reported: bool,
}

impl YoctiteViewPass {
    /// The view, choosing its clusters on the GPU, or `on_gpu` false on the CPU.
    pub(crate) fn new(source_type_id: TypeId, on_gpu: bool) -> Self {
        Self {
            source_type_id,
            pool: None,
            view: None,
            shader: None,
            built: Default::default(),
            placed: Default::default(),
            frame: 0,
            failed: false,
            on_gpu,
            culler: None,
            overflow_reported: false,
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

impl YoctiteViewPass {
    /// Keeps the pool in step with the scene's meshes: new surfaces built and uploaded, new
    /// mesh nodes placed, moved ones moved, gone ones taken out.
    fn sync(&mut self, server: &WgpuGraphicsServer, graph: &fyrox::scene::graph::Graph) {
        let device = &server.state.device;
        let queue = &server.state.queue;
        let pool = self.pool.get_or_insert_with(|| GeometryPool::new(device, 1 << 16));
        self.frame += 1;
        let frame = self.frame;
        let mut newly_built = (0, 0usize, 0.0f64);
        for (handle, node) in graph.pair_iter() {
            let Some(mesh) = node.cast::<Mesh>() else {
                continue;
            };
            // Skinned meshes stay on the renderer's own path.
            if !node.global_visibility() || mesh.surfaces().iter().any(|s| !s.bones().is_empty()) {
                continue;
            }
            let transform = node.global_transform();
            for (n, surface) in mesh.surfaces().iter().enumerate() {
                let data = surface.data().key();
                let mesh = match self.built.entry(data) {
                    Entry::Occupied(built) => built.get().mesh,
                    Entry::Vacant(place) => {
                        let (vertices, indices) = read_surface(surface);
                        let positions: Vec<[f32; 3]> = vertices.iter().map(|v| v.position).collect();
                        let started = Instant::now();
                        let dag = yoctite::build(&yoctite::Mesh::new(&positions, &indices), Default::default());
                        newly_built.0 += 1;
                        newly_built.1 += indices.len() / 3;
                        newly_built.2 += started.elapsed().as_secs_f64();
                        let mesh = pool.add_mesh(device, queue, &dag, &vertices);
                        place.insert(Built { mesh, dag: Arc::new(dag) }).mesh
                    }
                };
                match self.placed.get_mut(&(handle, n)) {
                    Some(placed) if placed.data == data => {
                        if placed.transform != transform {
                            pool.move_instance(queue, placed.instance, rows(&transform));
                            placed.transform = transform;
                        }
                        placed.seen = frame;
                    }
                    other => {
                        if let Some(old) = other {
                            pool.remove_instance(queue, old.instance);
                        }
                        let instance = pool.add_instance(device, queue, mesh, rows(&transform));
                        self.placed.insert(
                            (handle, n),
                            Placed {
                                instance,
                                data,
                                transform,
                                seen: frame,
                            },
                        );
                    }
                }
            }
        }
        self.placed.retain(|_, placed| {
            let keep = placed.seen == frame;
            if !keep {
                pool.remove_instance(queue, placed.instance);
            }
            keep
        });
        if newly_built.0 > 0 {
            let stats = pool.stats();
            Log::info(format!(
                "Yoctite: built {} surfaces ({} triangles) in {:.2} s; the pool holds {} meshes, {} instances, \
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
    }
}

/// The clusters to draw from where the camera is, chosen on the CPU, as instance slots and pool
/// clusters, and how many triangles they come to: the reference for the GPU's choice.
fn choose(
    built: &FxHashMap<u64, Built>,
    placed: &FxHashMap<(Handle<Node>, usize), Placed>,
    pool: &GeometryPool,
    view_projection: &Matrix4<f32>,
    eye: Vector3<f32>,
    threshold: f32,
) -> (Vec<[u32; 2]>, usize) {
    let mut list = Vec::new();
    let mut triangles = 0;
    let by_mesh: FxHashMap<MeshId, &Built> = built.values().map(|b| (b.mesh, b)).collect();
    for placed in placed.values() {
        let Some(record) = pool.instance(placed.instance) else {
            continue;
        };
        let Some(built) = by_mesh.get(&MeshId(record.mesh)) else {
            continue;
        };
        let Some(mesh) = pool.mesh(built.mesh) else {
            continue;
        };
        let t = &placed.transform;
        let scale = (0..3)
            .map(|c| t.column(c).xyz().norm())
            .fold(0.0, f32::max);
        let center = t.transform_point(&Point3::new(mesh.bounds[0], mesh.bounds[1], mesh.bounds[2]));
        if !in_view(view_projection, center.coords, mesh.bounds[3] * scale) {
            continue;
        }
        // In the mesh's own space, where its errors and bounds are.
        let Some(inverse) = t.try_inverse() else {
            continue;
        };
        let local = inverse.transform_point(&Point3::from(eye));
        for c in yoctite::select(&built.dag, [local.x, local.y, local.z], threshold) {
            triangles += built.dag.clusters[c as usize].triangles.len();
            list.push([placed.instance.0, mesh.first_cluster + c]);
        }
    }
    (list, triangles)
}

impl SceneRenderPass for YoctiteViewPass {
    fn on_hdr_render(&mut self, ctx: SceneRenderPassContext) -> Result<RenderPassStatistics, FrameworkError> {
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
        let started = Instant::now();
        self.sync(server, &ctx.scene.graph);
        let synced = started.elapsed().as_secs_f64();

        let position = &ctx.observer.position;
        let viewport = ctx.observer.viewport;
        // An error as big as a pixel: the view's height in radians over its height in pixels.
        let fov = 2.0 * (1.0 / position.projection_matrix[(1, 1)]).atan();
        let threshold = fov / viewport.h().max(1) as f32;
        let mut view_projection = [[0.0; 4]; 4];
        for (column, values) in position.view_projection_matrix.column_iter().zip(view_projection.iter_mut()) {
            values.copy_from_slice(column.as_slice());
        }
        let eye = position.translation;
        let (width, height) = (viewport.w() as u32, viewport.h() as u32);
        let view = self.view.get_or_insert_with(|| ClusterView::new(&server.state.device));
        let pool = self.pool.as_ref().expect("made by sync");
        let started = Instant::now();
        let (drawn, chosen) = if self.on_gpu {
            let culler = self.culler.get_or_insert_with(|| Culler::new(&server.state.device));
            let drawn = view.draw_culled(server, pool, culler, view_projection, eye.into(), threshold, width, height);
            // What the GPU chose, read back a frame or two late.
            let chosen = culler.stats().map(|stats| {
                if stats.overflowed && !self.overflow_reported {
                    Log::warn(format!(
                        "Yoctite view: {} clusters chosen, more than the draw list held; it holds {} from now on",
                        stats.clusters,
                        culler.capacity()
                    ));
                    self.overflow_reported = true;
                }
                (stats.clusters as usize, stats.triangles as usize)
            });
            (drawn, chosen)
        } else {
            let (list, triangles) = choose(&self.built, &self.placed, pool, &position.view_projection_matrix, eye, threshold);
            let drawn = view.draw(server, pool, view_projection, width, height, &list);
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
        let (clusters, triangles) = chosen.unwrap_or_default();
        if self.frame % STATS_EVERY == 1 {
            Log::info(format!(
                "Yoctite view: {} clusters, {} triangles, chosen on the {}; the pool kept up with the scene \
                 in {:.2} ms, choosing and drawing took the CPU {:.2} ms",
                clusters,
                triangles,
                if self.on_gpu { "GPU" } else { "CPU" },
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
}
