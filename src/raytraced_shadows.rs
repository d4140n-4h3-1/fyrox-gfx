//! Shadows traced against the scene's geometry, for every light.
//!
//! A shadow map is a picture of the scene from the light's point of view, and everything about it -
//! the stair-stepped edges, the peter-panning, the cascades - comes from it being a picture at a
//! fixed resolution. It also has to be drawn again every frame for every light, six times over for
//! a lamp, which is why a scene full of lamps can only afford shadows for the nearest few, and why
//! shadows come and go as the camera moves. Tracing asks the geometry directly instead: for each
//! pixel a light reaches, is there a triangle between it and the light? The answer is exact at any
//! distance, with no maps to size, no cascades to blend, and no bias to tune beyond keeping a
//! surface from shadowing itself - and its cost is the pixels lit, not the scene redrawn.
//!
//! This needs ray tracing hardware. [`TracedLightShadows`] is handed to the renderer as its
//! [`LightShadowTracer`]: the renderer asks it for a shadow mask for each light just before
//! drawing that light, and uses the mask in place of the light's shadow map, so the shadow darkens
//! that light alone. Without the hardware it returns no masks and the renderer's shadow maps are
//! used as before.
//!
//! Lights are given a size, and a few rays per pixel are spread across it, so a shadow's edge
//! softens where only part of the light is hidden - wider the further the shadow falls from what
//! casts it, as a real one does. The mask is then blurred a little, only across the same surface,
//! to smooth out the differences between neighboring pixels. One ray per pixel gives hard shadows
//! and skips the blur.
//!
//! Each mesh's triangles are put into the hardware's structure once, in the mesh's own space -
//! once for every mesh made from the same surface data, such as a maze's repeated tiles - and every
//! frame a copy of each is placed wherever its mesh is, so what moves casts its shadow from where
//! it is. A skinned mesh's triangles are skinned by its bones every frame, as the renderer draws
//! them, and refitted. Hidden meshes still cast shadows while they stand still, since a game hides
//! what the camera cannot see; once a mesh has moved, or if it is skinned, it casts none while
//! hidden, as with shadow maps.
//!
//! Glass stands in light's way too, but lets it through, taking on its colour: a lamp shone
//! through red glass casts red light past it. How much of each colour gets through is the glass's
//! tint, as strongly as its tint strength; glass told not to colour light
//! ([`crate::GlassMaterial::tints_light`]) is left out, and light goes through it as if it were not
//! there. The light is only coloured, not bent: where the glass would focus it, it does not.

use fyrox::{
    core::{
        algebra::{Matrix4, Point3, Vector3},
        color::Color,
        log::Log,
        pool::Handle,
        sstorage::ImmutableString,
    },
    fxhash::{FxHashMap, FxHashSet},
    graph::SceneGraph,
    graphics::{error::FrameworkError, gpu_texture::GpuTexture, server::GraphicsServer},
    material::{MaterialProperty, MaterialResource, MaterialResourceBinding},
    renderer::{
        bundle::LightSourceKind,
        traced_shadows::{LightShadowTraceContext, LightShadowTracer},
    },
    scene::{
        graph::Graph,
        mesh::{
            surface::Surface,
            buffer::{VertexAttributeUsage, VertexReadTrait},
            Mesh,
        },
        node::Node,
        Scene,
    },
};
use fyrox_graphics_wgpu::{
    raytracing::{
        RayTracedGeometry, RayTracedInstance, RayTracedScene, ShadowRayLight,
        ShadowRayParameters, ShadowTracer,
    },
    server::WgpuGraphicsServer,
};

/// How shadows are traced.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RayTracedShadows {
    /// How far a ray towards the sun reaches, in meters. Anything further away casts no shadow.
    /// Rays towards a lamp end at the lamp.
    pub reach: f32,
    /// How far off a surface a ray starts, in meters, so it does not shadow itself.
    pub bias: f32,
    /// Rays per pixel. One gives hard shadows; more give soft edges, at the cost of more rays.
    pub samples: u32,
    /// The radius of the glowing part of a point light, in meters. Bigger is softer.
    pub point_light_size: f32,
    /// The same for spot lights, which are usually small bulbs - a torch, a headlight.
    pub spot_light_size: f32,
    /// How big the sun looks, in degrees from its middle to its edge. The real one is about a
    /// quarter of a degree; a little more reads better at a game's scale.
    pub sun_angular_radius: f32,
}

impl Default for RayTracedShadows {
    fn default() -> Self {
        Self {
            reach: 200.0,
            bias: 0.02,
            samples: 4,
            point_light_size: 0.15,
            spot_light_size: 0.03,
            sun_angular_radius: 1.0,
        }
    }
}

impl RayTracedShadows {
    /// One ray per pixel: sharp shadows, and the cheapest.
    pub fn hard() -> Self {
        Self {
            samples: 1,
            ..Default::default()
        }
    }
}

/// Whether the graphics server can trace rays at all.
pub fn is_supported(server: &dyn GraphicsServer) -> bool {
    server
        .as_any()
        .downcast_ref::<WgpuGraphicsServer>()
        .is_some_and(|server| server.ray_tracing)
}

/// How a surface of `material` stands in the way of light: solid, glass that lets through so
/// much of a light's red, green and blue, or not at all.
#[derive(Debug, Clone, Copy, PartialEq)]
enum Blocks {
    Solid,
    Glass([f32; 3]),
}

/// How a surface of `material` stands in the way of light, if it does. Glass lets light through,
/// taking on its tint as much as its tint strength says, unless it is told not to colour light,
/// when it is left out as if it were not there. Any other material that draws into no shadow
/// maps is left out too.
fn blocks(material: &MaterialResource) -> Option<Blocks> {
    let material = material.state();
    let Some(material) = material.data_ref() else {
        return Some(Blocks::Solid);
    };
    if material.shader() == &crate::refraction::glass_shader() {
        let key = ImmutableString::new("properties");
        let Some(MaterialResourceBinding::PropertyGroup(group)) = material.bindings().get(&key)
        else {
            return None;
        };
        let float = |name: &str, default: f32| match group.property_ref(name) {
            Some(MaterialProperty::Float(v)) => *v,
            _ => default,
        };
        if float("tintsLight", 1.0) <= 0.0 {
            return None;
        }
        let strength = float("tintStrength", 0.0).clamp(0.0, 1.0);
        let tint = match group.property_ref("tint") {
            Some(MaterialProperty::Color(tint)) => *tint,
            _ => Color::WHITE,
        };
        let through = |c: u8| 1.0 - strength * (1.0 - c as f32 / 255.0);
        return Some(Blocks::Glass([through(tint.r), through(tint.g), through(tint.b)]));
    }
    let shader = material.shader().state();
    let Some(shader) = shader.data_ref() else {
        return Some(Blocks::Solid);
    };
    let opted_out = shader
        .definition
        .disabled_passes
        .iter()
        .any(|pass| pass == "PointShadow");
    (!opted_out).then_some(Blocks::Solid)
}

/// How far a mesh may drift, in meters, before it counts as having moved.
const MOVED: f32 = 1.0e-4;

/// Whether `a` and `b` put things in the same place.
fn same_place(a: &Matrix4<f32>, b: &Matrix4<f32>) -> bool {
    a.iter().zip(b.iter()).all(|(a, b)| (a - b).abs() <= MOVED)
}

/// `transform`'s top three rows, row by row, as an instance is placed with.
fn instance_transform(transform: &Matrix4<f32>) -> [f32; 12] {
    let mut out = [0.0; 12];
    for row in 0..3 {
        for column in 0..4 {
            out[row * 4 + column] = transform[(row, column)];
        }
    }
    out
}

/// `surface`'s vertex positions, three floats each: in its own space, or, skinned, where its
/// bones in `graph` put them in the world, as the renderer draws it.
fn positions(graph: &Graph, surface: &Surface) -> Vec<f32> {
    let bones: Vec<Matrix4<f32>> = surface
        .bones()
        .iter()
        .map(|&bone| {
            graph.try_get_node(bone).map_or(Matrix4::identity(), |bone| {
                bone.global_transform() * bone.inv_bind_pose_transform()
            })
        })
        .collect();
    let data = surface.data();
    let data = data.data_ref();
    let mut out = Vec::with_capacity(data.vertex_buffer.vertex_count() as usize * 3);
    for vertex in data.vertex_buffer.iter() {
        let Ok(position) = vertex.read_3_f32(VertexAttributeUsage::Position) else {
            break;
        };
        let point = if bones.is_empty() {
            position
        } else {
            let (Ok(which), Ok(weights)) = (
                vertex.read_4_u8(VertexAttributeUsage::BoneIndices),
                vertex.read_4_f32(VertexAttributeUsage::BoneWeight),
            ) else {
                break;
            };
            let mut world = Vector3::zeros();
            for (&bone, &weight) in which.iter().zip(weights.iter()) {
                if let Some(matrix) = bones.get(bone as usize) {
                    world += matrix.transform_point(&Point3::from(position)).coords * weight;
                }
            }
            world
        };
        out.extend_from_slice(&[point.x, point.y, point.z]);
    }
    out
}

/// `surface`'s triangles, as indices into the first `count` of its vertices: only whole ones.
fn triangles(surface: &Surface, count: usize) -> Vec<u32> {
    let data = surface.data();
    let data = data.data_ref();
    let count = count as u32;
    data.geometry_buffer
        .iter()
        .filter(|triangle| triangle.0.iter().all(|&i| i < count))
        .flat_map(|triangle| triangle.0)
        .collect()
}

/// Which geometry a surface is traced with: what it shares with every surface made from the same
/// data, or, skinned, its own.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum Shape {
    /// With whether it is glass: the same data could be drawn solid and as glass.
    Shared(u64, bool),
    Posed(Handle<Node>, usize),
}

/// Makes the renderer's shadows by tracing rays. [`crate::GraphicsEffects`] installs it.
pub struct TracedLightShadows {
    settings: RayTracedShadows,
    tracer: Option<ShadowTracer>,
    scene: Option<RayTracedScene>,
    /// Which scene it traces.
    scene_handle: Option<Handle<Scene>>,
    /// Each shape's geometry, built once, and posed geometry refitted every frame.
    geometry: FxHashMap<Shape, RayTracedGeometry>,
    /// Where each mesh was last frame, to tell when one moves; and the meshes that have.
    places: FxHashMap<Handle<Node>, Matrix4<f32>>,
    moved: FxHashSet<Handle<Node>>,
    unsupported_reported: bool,
}

impl std::fmt::Debug for TracedLightShadows {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TracedLightShadows")
            .field("settings", &self.settings)
            .field("built", &self.scene.is_some())
            .field("geometry", &self.geometry.len())
            .finish()
    }
}

impl TracedLightShadows {
    pub fn new(settings: RayTracedShadows) -> Self {
        Self {
            settings,
            tracer: None,
            scene: None,
            scene_handle: None,
            geometry: Default::default(),
            places: Default::default(),
            moved: Default::default(),
            unsupported_reported: false,
        }
    }
}

impl LightShadowTracer for TracedLightShadows {
    fn prepare(
        &mut self,
        server: &dyn GraphicsServer,
        scene_handle: Handle<Scene>,
        scene: &Scene,
    ) -> Result<(), FrameworkError> {
        let Some(server) = server.as_any().downcast_ref::<WgpuGraphicsServer>() else {
            return Ok(());
        };
        if !server.ray_tracing {
            if !self.unsupported_reported {
                Log::warn(
                    "Ray traced shadows: this graphics adapter has no ray tracing, \
                     leaving shadows to shadow maps.",
                );
                self.unsupported_reported = true;
            }
            return Ok(());
        }

        if self.scene_handle != Some(scene_handle) {
            self.scene = None;
            self.geometry.clear();
            self.places.clear();
            self.moved.clear();
            self.scene_handle = Some(scene_handle);
        }
        let graph = &scene.graph;

        // What each surface that casts shadows is traced with, where, and which surface it is; and
        // posed vertices.
        let mut placed: Vec<(Shape, [f32; 12], Handle<Node>, usize, Blocks)> = Vec::new();
        let mut posed: Vec<(Shape, Vec<f32>)> = Vec::new();
        let mut places = FxHashMap::default();
        for (handle, node) in graph.pair_iter() {
            let Some(mesh) = node.cast::<Mesh>() else {
                continue;
            };
            let transform = node.global_transform();
            if self
                .places
                .get(&handle)
                .is_some_and(|was| !same_place(was, &transform))
            {
                self.moved.insert(handle);
            }
            places.insert(handle, transform);
            if !node.cast_shadows() {
                continue;
            }
            let skinned = mesh.surfaces().iter().any(|s| !s.bones().is_empty());
            if (skinned || self.moved.contains(&handle)) && !node.global_visibility() {
                continue;
            }
            for (n, surface) in mesh.surfaces().iter().enumerate() {
                let Some(how) = blocks(surface.material()) else {
                    continue;
                };
                if surface.bones().is_empty() {
                    let glass = matches!(how, Blocks::Glass(_));
                    let shape = Shape::Shared(surface.data().key(), glass);
                    placed.push((shape, instance_transform(&transform), handle, n, how));
                } else {
                    let shape = Shape::Posed(handle, n);
                    posed.push((shape, positions(graph, surface)));
                    let at = instance_transform(&Matrix4::identity());
                    placed.push((shape, at, handle, n, how));
                }
            }
        }
        self.places = places;
        self.moved.retain(|handle| self.places.contains_key(handle));

        // Geometry for shapes seen for the first time, and posed ones whose vertices changed in
        // number.
        let posed_vertices = |shape: &Shape| posed.iter().find(|(s, _)| s == shape).map(|(_, v)| v);
        let mut built = 0;
        for &(shape, _, handle, n, how) in &placed {
            let fresh = match self.geometry.get(&shape) {
                None => true,
                Some(had) => posed_vertices(&shape)
                    .is_some_and(|v| v.len() != had.vertex_count() as usize * 3),
            };
            if !fresh {
                continue;
            }
            let Some(surface) = graph
                .try_get_node(handle)
                .ok()
                .and_then(|node| node.cast::<Mesh>())
                .and_then(|mesh| mesh.surfaces().get(n))
            else {
                continue;
            };
            let vertices = match posed_vertices(&shape) {
                Some(vertices) => vertices.clone(),
                None => positions(graph, surface),
            };
            let indices = triangles(surface, vertices.len() / 3);
            let updatable = matches!(shape, Shape::Posed(..));
            let geometry = match how {
                Blocks::Solid => server.build_ray_traced_geometry(&vertices, &indices, updatable)?,
                Blocks::Glass(_) => server.build_ray_traced_glass(&vertices, &indices, updatable)?,
            };
            if let Some(geometry) = geometry {
                self.geometry.insert(shape, geometry);
                built += 1;
            }
        }
        // Only what is still there.
        self.geometry
            .retain(|shape, _| placed.iter().any(|(s, ..)| s == shape));

        // Posed geometry, refitted where it is this frame.
        let updates: Vec<(&RayTracedGeometry, &[f32])> = posed
            .iter()
            .filter_map(|(shape, vertices)| {
                let geometry = self.geometry.get(shape)?;
                (vertices.len() == geometry.vertex_count() as usize * 3)
                    .then_some((geometry, vertices.as_slice()))
            })
            .collect();
        server.update_ray_traced_geometry(&updates)?;

        let instances: Vec<RayTracedInstance> = placed
            .iter()
            .filter_map(|(shape, transform, _, _, how)| {
                let geometry = self.geometry.get(shape)?;
                // Posed geometry built solid stays so even if its surface turns to glass.
                let lets_through = match how {
                    Blocks::Glass(through) if geometry.is_see_through() => *through,
                    _ => [0.0; 3],
                };
                Some(RayTracedInstance {
                    geometry,
                    transform: *transform,
                    lets_through,
                })
            })
            .collect();
        match self.scene.as_mut() {
            Some(traced) => server.update_ray_traced_instances(traced, &instances)?,
            None => self.scene = server.build_ray_traced_instances(&instances)?,
        }
        if built > 0 {
            if let Some(traced) = self.scene.as_ref() {
                Log::info(format!(
                    "Ray traced shadows: {} pieces of geometry, {} copies, {} triangles",
                    self.geometry.len(),
                    instances.len(),
                    traced.triangle_count()
                ));
            }
        }

        if self.tracer.is_none() {
            self.tracer = server.create_shadow_tracer();
        }
        Ok(())
    }

    fn trace(
        &mut self,
        ctx: LightShadowTraceContext,
    ) -> Result<Option<GpuTexture>, FrameworkError> {
        let Some(server) = ctx.server.as_any().downcast_ref::<WgpuGraphicsServer>() else {
            return Ok(None);
        };
        let (Some(scene), Some(tracer)) = (self.scene.as_ref(), self.tracer.as_mut()) else {
            return Ok(None);
        };

        let light = match ctx.light.kind {
            LightSourceKind::Directional { .. } => {
                // The renderer lights with the up vector as the direction towards the sun, so
                // the sun's light travels the other way.
                let Some(towards_sun) = ctx.light.up_vector.try_normalize(f32::EPSILON) else {
                    return Ok(None);
                };
                let direction = -towards_sun;
                ShadowRayLight::Directional {
                    direction: [direction.x, direction.y, direction.z],
                    reach: self.settings.reach,
                    angular_size: self.settings.sun_angular_radius.to_radians().tan(),
                }
            }
            LightSourceKind::Point { .. } | LightSourceKind::Spot { .. } => {
                let position = ctx.light.position;
                let size = if matches!(ctx.light.kind, LightSourceKind::Point { .. }) {
                    self.settings.point_light_size
                } else {
                    self.settings.spot_light_size
                };
                ShadowRayLight::Positional {
                    position: [position.x, position.y, position.z],
                    radius: ctx.light_radius,
                    size,
                }
            }
            LightSourceKind::Unknown => return Ok(None),
        };

        // The renderer's scissor box has its origin at the bottom left; the tracer's at the top.
        let scissor = ctx.scissor.map(|b| {
            let top = ctx.viewport.size.y - b.y - b.height;
            [
                b.x.max(0) as u32,
                top.max(0) as u32,
                b.width.max(0) as u32,
                b.height.max(0) as u32,
            ]
        });

        tracer.trace(
            server,
            scene,
            ctx.depth,
            ctx.normals,
            ShadowRayParameters {
                inverse_view_projection: matrix_to_array(&ctx.inv_view_projection),
                light,
                bias: self.settings.bias,
                // This backend stores render targets top row first.
                flip_v: true,
                scissor,
                samples: self.settings.samples,
            },
        )
    }
}

fn matrix_to_array(matrix: &Matrix4<f32>) -> [[f32; 4]; 4] {
    let mut out = [[0.0; 4]; 4];
    for (column, values) in matrix.column_iter().zip(out.iter_mut()) {
        values.copy_from_slice(column.as_slice());
    }
    out
}
