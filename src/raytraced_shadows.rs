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
//! The geometry is gathered once and again whenever meshes are added or removed, so anything that
//! moves casts its shadow from wherever it was when the scene was gathered.

use fyrox::{
    core::{algebra::Matrix4, log::Log, pool::Handle},
    graph::SceneGraph,
    graphics::{error::FrameworkError, gpu_texture::GpuTexture, server::GraphicsServer},
    material::MaterialResource,
    renderer::{
        bundle::LightSourceKind,
        traced_shadows::{LightShadowTraceContext, LightShadowTracer},
    },
    scene::{
        graph::Graph,
        mesh::{
            buffer::{VertexAttributeUsage, VertexReadTrait},
            Mesh,
        },
        Scene,
    },
};
use std::hash::{Hash, Hasher};
use fyrox_graphics_wgpu::{
    raytracing::{RayTracedScene, ShadowRayLight, ShadowRayParameters, ShadowTracer},
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

/// Whether a material draws into shadow maps. Materials that opt out - glass, which light passes
/// through - are left out of the traced geometry as well, or the glass housings in a ceiling would
/// shut in the light of the lamps mounted beneath them.
fn casts_shadows(material: &MaterialResource) -> bool {
    let material = material.state();
    let Some(material) = material.data_ref() else {
        return true;
    };
    let shader = material.shader().state();
    let Some(shader) = shader.data_ref() else {
        return true;
    };
    !shader
        .definition
        .disabled_passes
        .iter()
        .any(|pass| pass == "PointShadow")
}

/// Every shadow-casting triangle of every mesh of the scene, in world space, ready to be traced
/// against.
///
/// Hidden meshes are included: a game hides what the camera cannot see so it is not drawn, but it
/// is still there, and still stands between a light and whatever the camera does see.
fn collect_triangles(graph: &Graph) -> (Vec<f32>, Vec<u32>) {
    let mut vertices: Vec<f32> = Vec::new();
    let mut indices: Vec<u32> = Vec::new();
    for node in graph.linear_iter() {
        let Some(mesh) = node.cast::<Mesh>() else {
            continue;
        };
        if !node.cast_shadows() {
            continue;
        }
        let transform = node.global_transform();
        for surface in mesh.surfaces() {
            if !casts_shadows(surface.material()) {
                continue;
            }
            let data = surface.data();
            let data = data.data_ref();
            let base = (vertices.len() / 3) as u32;
            for vertex in data.vertex_buffer.iter() {
                let Ok(position) = vertex.read_3_f32(VertexAttributeUsage::Position) else {
                    continue;
                };
                let world = transform.transform_point(&position.into());
                vertices.extend_from_slice(&[world.x, world.y, world.z]);
            }
            for triangle in data.geometry_buffer.iter() {
                indices.extend_from_slice(&[
                    base + triangle[0],
                    base + triangle[1],
                    base + triangle[2],
                ]);
            }
        }
    }
    (vertices, indices)
}

/// Makes the renderer's shadows by tracing rays. [`crate::GraphicsEffects`] installs it.
pub struct TracedLightShadows {
    settings: RayTracedShadows,
    tracer: Option<ShadowTracer>,
    scene: Option<RayTracedScene>,
    /// Which scene the structure was built from, and which meshes it had, so it is rebuilt when
    /// the scene changes.
    built_from: Option<(Handle<Scene>, u64)>,
    unsupported_reported: bool,
}

impl std::fmt::Debug for TracedLightShadows {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TracedLightShadows")
            .field("settings", &self.settings)
            .field("built", &self.scene.is_some())
            .finish()
    }
}

impl TracedLightShadows {
    pub fn new(settings: RayTracedShadows) -> Self {
        Self {
            settings,
            tracer: None,
            scene: None,
            built_from: None,
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

        // Which meshes there are, as one number. Counting them is not enough: a level torn down
        // and replaced by another can have as many meshes as the last, in other places.
        let meshes = {
            let mut hasher = fyrox::fxhash::FxHasher64::default();
            for (handle, node) in scene.graph.pair_iter() {
                if node.cast::<Mesh>().is_some() {
                    handle.hash(&mut hasher);
                }
            }
            hasher.finish()
        };
        if self.built_from != Some((scene_handle, meshes)) {
            let (vertices, indices) = collect_triangles(&scene.graph);
            self.scene = server.build_ray_traced_scene(&vertices, &indices)?;
            self.built_from = Some((scene_handle, meshes));
            if let Some(scene) = self.scene.as_ref() {
                Log::info(format!(
                    "Ray traced shadows: {} triangles in the acceleration structure",
                    scene.triangle_count()
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
