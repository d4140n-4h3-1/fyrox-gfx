//! Graphics improvements for Fyrox that live outside the engine.
//!
//! Add [`GraphicsEffects`] to the executor next to the game's own plugin, then use the effects:
//!
//! - [`refraction`]: glass that bends and tints whatever is behind it. Build a material with
//!   [`GlassMaterial`] and put it on any mesh.
//! - [`shadows`]: softer shadow edges, applied to the renderer's quality settings on startup.
//!   Pass [`GraphicsEffects::without_soft_shadows`] to keep the game's own settings.
//! - [`antialiasing`]: temporal anti-aliasing ([`temporal`]), on by default, or the renderer's
//!   FXAA.
//! - [`occlusion`]: how far ambient occlusion reaches, which is what keeps corners from going
//!   flat. On by default, at the engine's own strength.
//! - [`reflections`]: ray-marched against the depth buffer. Off by default.
//! - [`moving`]: what in the scene moves, which the game tells the effects every frame so that
//!   temporal anti-aliasing does not leave a ghost behind it.
//! - [`shadows::ShadowBudget`]: keeps shadow maps for the lights nearest the camera only, which is
//!   what makes a scene full of small lamps affordable. On by default, and idle while shadows are
//!   traced, since then there are no shadow maps to budget.
//! - [`shadows::LightBudget`]: the same for the lights themselves. Off by default, because a light
//!   going out is visible when it lit something the player can see.
//!
//! The crate works with either engine backend - OpenGL or wgpu (Vulkan) - and follows the one the
//! game was built with.
//!
//! # About ray tracing
//!
//! [`raytraced_shadows`] (feature `raytracing`) traces the shadows of every light against the
//! scene's geometry using the graphics card's ray tracing hardware, through the engine's wgpu
//! backend. It replaces the renderer's shadow maps, and leaves them in place on hardware without
//! ray tracing.
//!
//! [`reflections`] traces rays too, but against the depth buffer: the scene as the camera sees it.
//! That is what can be done from outside the engine, and it is what most games ship. It cannot
//! show what the camera cannot - a reflection of something behind the camera or around a corner
//! has nothing to trace against.
//!
//! It is cheaper than tracing geometry and needs no special hardware, which is why it is the one
//! that is on for games that want reflections everywhere.

pub mod antialiasing;
pub mod moving;
pub mod occlusion;
#[cfg(feature = "raytracing")]
pub mod raytraced_shadows;
pub mod reflections;
pub mod refraction;
mod scene_copy;
pub mod shadows;
pub mod temporal;

pub use antialiasing::AntiAliasing;
pub use moving::{MovingThing, MovingThings};
pub use occlusion::AmbientOcclusion;
#[cfg(feature = "raytracing")]
pub use raytraced_shadows::RayTracedShadows;
pub use reflections::Reflections;
pub use refraction::{replace_materials, GlassMaterial};
pub use shadows::{LightBudget, ShadowBudget, SoftShadows};

use fyrox::{
    core::{reflect::prelude::*, visitor::prelude::*},
    engine::GraphicsContext,
    plugin::{error::GameResult, Plugin, PluginContext},
};
use refraction::RefractionPass;
use std::{any::TypeId, cell::RefCell, rc::Rc};

/// Installs the crate's render passes into the renderer.
#[derive(Debug, Visit, Reflect)]
#[reflect(non_cloneable, type_uuid = "b6e3f1f2-8d0a-4c1e-a1f7-2c5b9e4d7a10")]
pub struct GraphicsEffects {
    #[visit(skip)]
    #[reflect(hidden)]
    refraction: Option<Rc<RefCell<RefractionPass>>>,
    #[visit(skip)]
    #[reflect(hidden)]
    soft_shadows: Option<SoftShadows>,
    #[visit(skip)]
    #[reflect(hidden)]
    anti_aliasing: Option<AntiAliasing>,
    #[visit(skip)]
    #[reflect(hidden)]
    shadow_budget: Option<ShadowBudget>,
    #[visit(skip)]
    #[reflect(hidden)]
    light_budget: Option<LightBudget>,
    #[visit(skip)]
    #[reflect(hidden)]
    ambient_occlusion: Option<AmbientOcclusion>,
    #[visit(skip)]
    #[reflect(hidden)]
    reflections: Option<Reflections>,
    #[visit(skip)]
    #[reflect(hidden)]
    reflection_pass: Option<Rc<RefCell<reflections::ReflectionPass>>>,
    #[visit(skip)]
    #[reflect(hidden)]
    temporal_pass: Option<Rc<RefCell<temporal::TemporalPass>>>,
    /// What moves, as the game says.
    #[visit(skip)]
    #[reflect(hidden)]
    moving: MovingThings,
    /// Frames drawn with temporal anti-aliasing, which picks each frame's shift of the view.
    #[visit(skip)]
    #[reflect(hidden)]
    temporal_frame: u64,
    #[cfg(feature = "raytracing")]
    #[visit(skip)]
    #[reflect(hidden)]
    ray_traced_shadows: Option<RayTracedShadows>,
    /// Whether the renderer's shadows are being traced, which leaves the shadow budget with
    /// nothing to do.
    #[visit(skip)]
    #[reflect(hidden)]
    shadows_traced: bool,
}

impl Default for GraphicsEffects {
    fn default() -> Self {
        Self {
            refraction: None,
            soft_shadows: Some(SoftShadows::default()),
            anti_aliasing: Some(AntiAliasing::default()),
            shadow_budget: Some(ShadowBudget::default()),
            // Off by default: switching a light off is visible when it lights something the
            // player can see, so a game should choose the number itself.
            light_budget: None,
            ambient_occlusion: Some(AmbientOcclusion::default()),
            reflections: None,
            reflection_pass: None,
            temporal_pass: None,
            moving: MovingThings::default(),
            temporal_frame: 0,
            #[cfg(feature = "raytracing")]
            ray_traced_shadows: None,
            shadows_traced: false,
        }
    }
}

impl GraphicsEffects {
    /// The list of what moves in the scene, for the game to keep up to date every frame - see
    /// [`moving`]. The effects share it, so a copy taken before the plugin is handed to the
    /// executor stays connected.
    pub fn moving_things(&self) -> MovingThings {
        self.moving.clone()
    }

    /// Leaves shadow settings as the game set them.
    pub fn without_soft_shadows(mut self) -> Self {
        self.soft_shadows = None;
        self
    }

    /// Uses the given shadow settings instead of the default ones.
    pub fn with_soft_shadows(mut self, shadows: SoftShadows) -> Self {
        self.soft_shadows = Some(shadows);
        self
    }

    /// Lets every light draw shadows, however many there are.
    pub fn without_shadow_budget(mut self) -> Self {
        self.shadow_budget = None;
        self
    }

    /// Uses the given shadow budget instead of the default one.
    pub fn with_shadow_budget(mut self, budget: ShadowBudget) -> Self {
        self.shadow_budget = Some(budget);
        self
    }

    /// Lets every light in the scene light it, however many there are. This is the default.
    pub fn without_light_budget(mut self) -> Self {
        self.light_budget = None;
        self
    }

    /// Uses the given light budget instead of the default one.
    pub fn with_light_budget(mut self, budget: LightBudget) -> Self {
        self.light_budget = Some(budget);
        self
    }

    /// Traces the shadows of every light against the scene's geometry, where the hardware can,
    /// in place of the renderer's shadow maps. Off by default.
    #[cfg(feature = "raytracing")]
    pub fn with_ray_traced_shadows(mut self, shadows: RayTracedShadows) -> Self {
        self.ray_traced_shadows = Some(shadows);
        self
    }

    /// Traces reflections on surfaces facing upwards. Off by default.
    pub fn with_reflections(mut self, reflections: Reflections) -> Self {
        self.reflections = Some(reflections);
        self
    }

    /// Leaves ambient occlusion as the game set it.
    pub fn without_ambient_occlusion(mut self) -> Self {
        self.ambient_occlusion = None;
        self
    }

    /// Uses the given ambient occlusion instead of the default.
    pub fn with_ambient_occlusion(mut self, occlusion: AmbientOcclusion) -> Self {
        self.ambient_occlusion = Some(occlusion);
        self
    }

    /// Leaves anti-aliasing as the game set it.
    pub fn without_anti_aliasing(mut self) -> Self {
        self.anti_aliasing = None;
        self
    }

    /// Uses the given anti-aliasing setting instead of the default one.
    pub fn with_anti_aliasing(mut self, anti_aliasing: AntiAliasing) -> Self {
        self.anti_aliasing = Some(anti_aliasing);
        self
    }
}

impl PartialEq for GraphicsEffects {
    fn eq(&self, other: &Self) -> bool {
        match (&self.refraction, &other.refraction) {
            (Some(a), Some(b)) => Rc::ptr_eq(a, b),
            (None, None) => true,
            _ => false,
        }
    }
}

impl Plugin for GraphicsEffects {
    fn on_graphics_context_initialized(&mut self, ctx: PluginContext) -> GameResult {
        if let GraphicsContext::Initialized(graphics_context) = ctx.graphics_context {
            let pass = Rc::new(RefCell::new(RefractionPass::new(TypeId::of::<Self>())));
            graphics_context.renderer.add_render_pass(pass.clone());
            self.refraction = Some(pass);
            if let Some(shadows) = self.soft_shadows {
                shadows.apply(&mut graphics_context.renderer);
            }
            let mut temporal = false;
            if let Some(mut anti_aliasing) = self.anti_aliasing {
                // Temporal anti-aliasing has a wgpu shader only; on OpenGL, FXAA stands in.
                if anti_aliasing.temporal && !refraction::engine_uses_wgsl() {
                    anti_aliasing = AntiAliasing::fxaa();
                }
                anti_aliasing.apply(&mut graphics_context.renderer);
                temporal = anti_aliasing.temporal;
            }
            if let Some(occlusion) = self.ambient_occlusion {
                occlusion.apply(&mut graphics_context.renderer);
            }
            #[cfg(feature = "raytracing")]
            if let Some(shadows) = self.ray_traced_shadows {
                let renderer = &mut graphics_context.renderer;
                // Only where it can work: otherwise shadow maps, and their budget, stay in charge.
                self.shadows_traced = raytraced_shadows::is_supported(renderer.graphics_server());
                if self.shadows_traced {
                    renderer.set_light_shadow_tracer(Some(Box::new(
                        raytraced_shadows::TracedLightShadows::new(shadows),
                    )));
                }
            }
            if let Some(reflections) = self.reflections {
                // After the glass pass, so glass shows up in reflections too.
                let pass = Rc::new(RefCell::new(reflections::ReflectionPass::new(
                    TypeId::of::<Self>(),
                    reflections,
                )));
                graphics_context.renderer.add_render_pass(pass.clone());
                self.reflection_pass = Some(pass);
            }
            if temporal {
                // Last, so it smooths everything drawn before it: glass and reflections too.
                let pass = Rc::new(RefCell::new(temporal::TemporalPass::new(
                    TypeId::of::<Self>(),
                    self.moving.clone(),
                )));
                graphics_context.renderer.add_render_pass(pass.clone());
                self.temporal_pass = Some(pass);
            }
        }
        Ok(())
    }

    fn update(&mut self, ctx: &mut PluginContext) -> GameResult {
        if self.temporal_pass.is_some() {
            if let GraphicsContext::Initialized(graphics_context) = &mut *ctx.graphics_context {
                self.temporal_frame += 1;
                let jitter = temporal::jitter(self.temporal_frame);
                graphics_context.renderer.set_projection_jitter(jitter);
                if let Some(pass) = self.temporal_pass.as_ref() {
                    pass.borrow_mut().set_jitter(jitter);
                }
            }
        }

        // Traced shadows come from the lights' cast_shadows flag too, so the budget would switch
        // them off for all but the nearest lights.
        let mut shadow_budget = self.shadow_budget.as_mut().filter(|_| !self.shadows_traced);
        if shadow_budget.is_some() || self.light_budget.is_some() {
            let handles: Vec<_> = ctx.scenes.pair_iter().map(|(handle, _)| handle).collect();
            for handle in &handles {
                let Ok(scene) = ctx.scenes.try_get_mut(*handle) else {
                    continue;
                };
                // Lights first: a light that is off does not need a shadow map either.
                if let Some(budget) = self.light_budget.as_mut() {
                    budget.apply(*handle, scene);
                }
                if let Some(budget) = shadow_budget.as_deref_mut() {
                    budget.apply(*handle, scene);
                }
            }
            if let Some(budget) = self.light_budget.as_mut() {
                budget.forget_missing(|scene| handles.contains(&scene));
            }
            if let Some(budget) = shadow_budget {
                budget.forget_missing(|scene| handles.contains(&scene));
            }
        }
        Ok(())
    }

    fn on_graphics_context_destroyed(&mut self, ctx: PluginContext) -> GameResult {
        // The renderer is gone with the context; the pass holds GPU objects of that renderer.
        let _ = ctx;
        self.refraction = None;
        self.reflection_pass = None;
        self.temporal_pass = None;
        Ok(())
    }

    fn on_deinit(&mut self, ctx: PluginContext) -> GameResult {
        if let GraphicsContext::Initialized(graphics_context) = ctx.graphics_context {
            if let Some(pass) = self.refraction.take() {
                graphics_context.renderer.remove_render_pass(pass);
            }
            if let Some(pass) = self.reflection_pass.take() {
                graphics_context.renderer.remove_render_pass(pass);
            }
            if let Some(pass) = self.temporal_pass.take() {
                graphics_context.renderer.remove_render_pass(pass);
                graphics_context
                    .renderer
                    .set_projection_jitter(Default::default());
            }
            if self.shadows_traced {
                graphics_context.renderer.set_light_shadow_tracer(None);
                self.shadows_traced = false;
            }
        }
        Ok(())
    }
}
