(
    name: "FyroxGfxReflections",

    resources: [
        (
            // The frame before reflections were added.
            name: "sceneColor",
            kind: Texture(kind: Sampler2D, fallback: Black),
            binding: 0
        ),
        (
            name: "sceneDepth",
            kind: Texture(kind: DepthSampler2D, fallback: White),
            binding: 1
        ),
        (
            name: "sceneNormal",
            kind: Texture(kind: Sampler2D, fallback: Black),
            binding: 2
        ),
        (
            name: "properties",
            kind: PropertyGroup([
                (name: "worldViewProjection", kind: Matrix4()),
                (name: "viewProjection", kind: Matrix4()),
                (name: "inverseViewProjection", kind: Matrix4()),
                (name: "cameraPosition", kind: Vector3()),
                (name: "screenSize", kind: Vector2()),
                // How far a reflected ray travels, in meters.
                (name: "reach", kind: Float(value: 12.0)),
                // How many steps it takes to get there. More steps cost more and miss less.
                (name: "steps", kind: Int(value: 24)),
                // How far behind a surface the ray may pass and still count as hitting it.
                (name: "thickness", kind: Float(value: 0.4)),
                // How much of the reflection is kept.
                (name: "strength", kind: Float(value: 0.35)),
                // Only surfaces facing at least this far upwards reflect; 1 is straight up.
                (name: "minUpwards", kind: Float(value: 0.7)),
            ]),
            binding: 0
        ),
    ],

    passes: [
        (
            name: "Primary",
            draw_parameters: DrawParameters(
                cull_face: None,
                color_write: ColorMask(
                    red: true,
                    green: true,
                    blue: true,
                    alpha: true,
                ),
                depth_write: false,
                stencil_test: None,
                depth_test: None,
                // Mixed into what is already there, by how much of the surface reflects.
                blend: Some(BlendParameters(
                    func: BlendFunc(
                        sfactor: SrcAlpha,
                        dfactor: OneMinusSrcAlpha,
                        alpha_sfactor: SrcAlpha,
                        alpha_dfactor: OneMinusSrcAlpha,
                    ),
                    equation: BlendEquation(
                        rgb: Add,
                        alpha: Add
                    )
                )),
                stencil_op: StencilOp(
                    fail: Keep,
                    zfail: Keep,
                    zpass: Keep,
                    write_mask: 0xFFFF_FFFF,
                ),
                scissor_box: None
            ),

            vertex_shader:
                r#"
                    struct VertexInput {
                        @location(0) vertexPosition: vec3f,
                        @location(1) vertexTexCoord: vec2f,
                    };

                    struct VertexOutput {
                        @builtin(position) position: vec4f,
                        @location(0) texCoord: vec2f,
                    };

                    @vertex fn vs_main(input: VertexInput) -> VertexOutput {
                        var output: VertexOutput;
                        output.texCoord = input.vertexTexCoord;
                        output.position = properties.worldViewProjection * vec4f(input.vertexPosition, 1.0);
                        return output;
                    }
                "#,

            fragment_shader:
                r#"
                    // Where a world-space point lands on screen: texture coordinates, and the
                    // depth the depth buffer would hold for it.
                    fn toScreen(point: vec3f) -> vec3f {
                        let clip = properties.viewProjection * vec4f(point, 1.0);
                        let ndc = clip.xyz / clip.w;
                        // Render targets are stored top row first here, so v runs against y.
                        return vec3f(ndc.x * 0.5 + 0.5, 0.5 - ndc.y * 0.5, ndc.z * 0.5 + 0.5);
                    }

                    fn worldAt(uv: vec2f, depth: f32) -> vec3f {
                        return S_UnProject(vec3f(uv, depth), properties.inverseViewProjection);
                    }

                    @fragment fn fs_main(@builtin(position) fragCoord: vec4f) -> @location(0) vec4f {
                        let uv = fragCoord.xy / properties.screenSize;
                        let depth = textureSample(sceneDepth_tex, sceneDepth_samp, uv);
                        // Nothing was drawn here, so there is nothing to reflect in.
                        if (depth >= 1.0) {
                            return vec4f(0.0);
                        }

                        let normal = normalize(textureSample(sceneNormal_tex, sceneNormal_samp, uv).xyz * 2.0 - 1.0);
                        // Reflections are kept to floors and other surfaces facing upwards: a
                        // screen-space ray has nothing to go on where the reflected view leaves
                        // the screen, which is most of the time on walls facing the camera.
                        let upwards = normal.y;
                        if (upwards < properties.minUpwards) {
                            return vec4f(0.0);
                        }

                        let position = worldAt(uv, depth);
                        let toCamera = normalize(properties.cameraPosition - position);
                        let ray = reflect(-toCamera, normal);
                        if (dot(ray, normal) <= 0.0) {
                            return vec4f(0.0);
                        }

                        let stepCount = max(properties.steps, 1);
                        let stepLength = properties.reach / f32(stepCount);
                        // Start a step out, so a surface never reflects itself.
                        var travelled = stepLength;
                        var hitUv = vec2f(0.0);
                        var hit = false;

                        for (var i: i32 = 0; i < stepCount; i++) {
                            let sample_point = position + ray * travelled;
                            let screen = toScreen(sample_point);
                            if (screen.x < 0.0 || screen.x > 1.0 || screen.y < 0.0 || screen.y > 1.0 || screen.z > 1.0) {
                                break;
                            }
                            let sceneDepth = textureSample(sceneDepth_tex, sceneDepth_samp, screen.xy);
                            if (sceneDepth < screen.z) {
                                // The ray has gone behind something. If it is only just behind,
                                // that something is what it hit; if it is far behind, the ray
                                // passed behind a foreground object and there is nothing to show.
                                let scene_position = worldAt(screen.xy, sceneDepth);
                                if (distance(scene_position, sample_point) < properties.thickness) {
                                    hitUv = screen.xy;
                                    hit = true;
                                }
                                break;
                            }
                            travelled += stepLength;
                        }

                        if (!hit) {
                            return vec4f(0.0);
                        }

                        // Fade out where the reflection runs off the screen, and where it is seen
                        // head on - a floor reflects most at a grazing angle.
                        let edge = min(
                            min(hitUv.x, 1.0 - hitUv.x),
                            min(hitUv.y, 1.0 - hitUv.y)
                        );
                        let edgeFade = smoothstep(0.0, 0.15, edge);
                        let grazing = pow(1.0 - clamp(dot(toCamera, normal), 0.0, 1.0), 2.0);
                        let distanceFade = 1.0 - clamp(travelled / properties.reach, 0.0, 1.0);

                        let amount = properties.strength * edgeFade * distanceFade * mix(0.25, 1.0, grazing);
                        let color = textureSample(sceneColor_tex, sceneColor_samp, hitUv).rgb;
                        return vec4f(color, amount);
                    }
                "#,
        )
    ]
)
