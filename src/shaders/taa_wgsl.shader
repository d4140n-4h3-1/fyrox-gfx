(
    name: "FyroxGfxTemporalAntiAliasing",

    resources: [
        (
            // This frame, drawn with this frame's sub-pixel shift.
            name: "currentColor",
            kind: Texture(kind: Sampler2D, fallback: Black),
            binding: 0
        ),
        (
            // Every frame so far, blended.
            name: "historyColor",
            kind: Texture(kind: Sampler2D, fallback: Black),
            binding: 1
        ),
        (
            name: "sceneDepth",
            kind: Texture(kind: DepthSampler2D, fallback: White),
            binding: 2
        ),
        (
            name: "properties",
            kind: PropertyGroup([
                (name: "worldViewProjection", kind: Matrix4()),
                (name: "inverseViewProjection", kind: Matrix4()),
                // Where the camera was last frame, to find where each point was on screen then.
                (name: "previousViewProjection", kind: Matrix4()),
                (name: "screenSize", kind: Vector2()),
                // Where each pixel's content is in this frame, which was drawn shifted by a
                // fraction of a pixel, as an offset in texture coordinates.
                (name: "jitterUv", kind: Vector2()),
                // How much of this frame goes into the result; the rest is history.
                (name: "blend", kind: Float(value: 0.1)),
                // Start over from this frame alone: the view jumped, and the history shows
                // somewhere else.
                (name: "reset", kind: Bool(value: false)),
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
                blend: None,
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
                    // Blending bright and dark pixels as they are lets one bright pixel - a lamp
                    // panel, many times brighter than a wall - dominate the average, and edges
                    // against lamps would still flicker. The blending is done on colors squeezed
                    // into 0..1 instead, and the result stretched back.
                    fn squeeze(c: vec3f) -> vec3f {
                        return c / (1.0 + max(c.r, max(c.g, c.b)));
                    }

                    fn unsqueeze(c: vec3f) -> vec3f {
                        return c / max(1.0 - max(c.r, max(c.g, c.b)), 1.0e-4);
                    }

                    fn current(uv: vec2f) -> vec3f {
                        return squeeze(textureSampleLevel(currentColor_tex, currentColor_samp, uv, 0.0).rgb);
                    }

                    @fragment fn fs_main(@builtin(position) fragCoord: vec4f) -> @location(0) vec4f {
                        let texel = 1.0 / properties.screenSize;
                        let uv = fragCoord.xy * texel;
                        // The result is lined up with the view as it would be without this
                        // frame's shift, so it holds still while the shift changes. This frame's
                        // color for the pixel is where the shift moved it.
                        let here = current(uv + properties.jitterUv);

                        // The colors around this pixel this frame (the samples at pixel middles
                        // are exact, filtering or not). Whatever the history says
                        // must lie among them: where it does not, the history shows something
                        // that has moved away or come into view, and following it would leave a
                        // ghost behind.
                        var low = here;
                        var high = here;
                        for (var y = -1; y <= 1; y++) {
                            for (var x = -1; x <= 1; x++) {
                                let near = current(uv + vec2f(f32(x), f32(y)) * texel);
                                low = min(low, near);
                                high = max(high, near);
                            }
                        }

                        // Where this pixel's point was on screen last frame. Nothing in the
                        // maze moves, so the depth buffer and the two camera positions are all
                        // it takes.
                        let depth = textureSampleLevel(sceneDepth_tex, sceneDepth_samp, uv + properties.jitterUv, 0);
                        let world = S_UnProject(vec3f(uv, depth), properties.inverseViewProjection);
                        let previous = properties.previousViewProjection * vec4f(world, 1.0);
                        let previousNdc = previous.xy / previous.w;
                        // Render targets are stored top row first here, so v runs against y.
                        let previousUv = vec2f(previousNdc.x * 0.5 + 0.5, 0.5 - previousNdc.y * 0.5);
                        let history = clamp(
                            squeeze(textureSampleLevel(historyColor_tex, historyColor_samp, previousUv, 0.0).rgb),
                            low,
                            high,
                        );

                        let offScreen = any(previousUv < vec2f(0.0)) || any(previousUv > vec2f(1.0));
                        let fresh = properties.reset != 0u || offScreen;
                        // The faster the view moves, the more this frame counts. Not everything
                        // stays put in the world - the flashlight's beam turns with the camera -
                        // and the history of what moved with the view would trail behind it.
                        // Fast motion hides aliasing anyway; it is smoothed again as it slows.
                        let motion = length((previousUv - uv) * properties.screenSize);
                        let blend = mix(properties.blend, 0.5, clamp(motion / 10.0, 0.0, 1.0));
                        let result = select(mix(history, here, blend), here, fresh);
                        return vec4f(unsqueeze(result), 1.0);
                    }
                "#,
        )
    ]
)
