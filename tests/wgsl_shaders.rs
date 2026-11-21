//! Every wgpu shader of the crate, assembled as the engine's wgpu backend assembles it, is valid
//! WGSL: one that is not otherwise fails only when its effect is first drawn, in a running game.

use fyrox::material::shader::Shader;
use fyrox_graphics_wgpu::{program::assemble_wgsl, vertex_depth::remap_vertex_depth};

#[test]
fn every_wgpu_shader_is_valid_wgsl() {
    let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("src/shaders");
    let mut shaders = 0;
    let mut stages = 0;
    for entry in std::fs::read_dir(&dir).unwrap() {
        let path = entry.unwrap().path();
        let is_wgsl = path
            .file_name()
            .and_then(|name| name.to_str())
            .is_some_and(|name| name.ends_with("_wgsl.shader"));
        if !is_wgsl {
            continue;
        }
        let text = std::fs::read_to_string(&path).unwrap();
        let shader = Shader::from_string(&text).unwrap_or_else(|e| panic!("{path:?}: {e}"));
        let resources = &shader.definition.resources;
        for pass in &shader.definition.passes {
            let vertex = remap_vertex_depth(&pass.vertex_shader.0).unwrap_or_else(|| pass.vertex_shader.0.clone());
            for (stage, source) in [("vertex", vertex), ("fragment", pass.fragment_shader.0.clone())] {
                let wgsl = assemble_wgsl(&source, resources);
                let module = naga::front::wgsl::parse_str(&wgsl)
                    .unwrap_or_else(|e| panic!("{path:?} {} {stage}: {}", pass.name, e.emit_to_string(&wgsl)));
                naga::valid::Validator::new(naga::valid::ValidationFlags::all(), naga::valid::Capabilities::all())
                    .validate(&module)
                    .unwrap_or_else(|e| panic!("{path:?} {} {stage}: {e:?}", pass.name));
                stages += 1;
            }
        }
        shaders += 1;
    }
    assert_eq!(shaders, 6, "the crate's six wgpu shaders");
    assert!(stages >= 12, "only {stages} stages found");
}
