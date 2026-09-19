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
            kind: Texture(kind: Sampler2D, fallback: White),
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
                    layout(location = 0) in vec3 vertexPosition;
                    layout(location = 1) in vec2 vertexTexCoord;

                    out vec2 texCoord;

                    void main()
                    {
                        texCoord = vertexTexCoord;
                        gl_Position = properties.worldViewProjection * vec4(vertexPosition, 1.0);
                    }
                "#,

            fragment_shader:
                r#"
                    out vec4 FragColor;

                    in vec2 texCoord;

                    // Where a world-space point lands on screen: texture coordinates, and the
                    // depth the depth buffer would hold for it.
                    vec3 toScreen(vec3 point) {
                        vec4 clip = properties.viewProjection * vec4(point, 1.0);
                        vec3 ndc = clip.xyz / clip.w;
                        return vec3(ndc.x * 0.5 + 0.5, ndc.y * 0.5 + 0.5, ndc.z * 0.5 + 0.5);
                    }

                    vec3 worldAt(vec2 uv, float depth) {
                        return S_UnProject(vec3(uv, depth), properties.inverseViewProjection);
                    }

                    void main()
                    {
                        vec2 uv = gl_FragCoord.xy / properties.screenSize;
                        float depth = texture(sceneDepth, uv).r;
                        // Nothing was drawn here, so there is nothing to reflect in.
                        if (depth >= 1.0) {
                            FragColor = vec4(0.0);
                            return;
                        }

                        vec3 normal = normalize(texture(sceneNormal, uv).xyz * 2.0 - 1.0);
                        // Reflections are kept to floors and other surfaces facing upwards: a
                        // screen-space ray has nothing to go on where the reflected view leaves
                        // the screen, which is most of the time on walls facing the camera.
                        if (normal.y < properties.minUpwards) {
                            FragColor = vec4(0.0);
                            return;
                        }

                        vec3 position = worldAt(uv, depth);
                        vec3 toCamera = normalize(properties.cameraPosition - position);
                        vec3 ray = reflect(-toCamera, normal);
                        if (dot(ray, normal) <= 0.0) {
                            FragColor = vec4(0.0);
                            return;
                        }

                        int stepCount = max(properties.steps, 1);
                        float stepLength = properties.reach / float(stepCount);
                        // Start a step out, so a surface never reflects itself.
                        float travelled = stepLength;
                        vec2 hitUv = vec2(0.0);
                        bool hit = false;

                        for (int i = 0; i < stepCount; ++i) {
                            vec3 samplePoint = position + ray * travelled;
                            vec3 screen = toScreen(samplePoint);
                            if (screen.x < 0.0 || screen.x > 1.0 || screen.y < 0.0 || screen.y > 1.0 || screen.z > 1.0) {
                                break;
                            }
                            float sceneDepthValue = texture(sceneDepth, screen.xy).r;
                            if (sceneDepthValue < screen.z) {
                                // The ray has gone behind something. If it is only just behind,
                                // that something is what it hit; if it is far behind, the ray
                                // passed behind a foreground object and there is nothing to show.
                                vec3 scenePosition = worldAt(screen.xy, sceneDepthValue);
                                if (distance(scenePosition, samplePoint) < properties.thickness) {
                                    hitUv = screen.xy;
                                    hit = true;
                                }
                                break;
                            }
                            travelled += stepLength;
                        }

                        if (!hit) {
                            FragColor = vec4(0.0);
                            return;
                        }

                        // Fade out where the reflection runs off the screen, and where it is seen
                        // head on - a floor reflects most at a grazing angle.
                        float edge = min(min(hitUv.x, 1.0 - hitUv.x), min(hitUv.y, 1.0 - hitUv.y));
                        float edgeFade = smoothstep(0.0, 0.15, edge);
                        float grazing = pow(1.0 - clamp(dot(toCamera, normal), 0.0, 1.0), 2.0);
                        float distanceFade = 1.0 - clamp(travelled / properties.reach, 0.0, 1.0);

                        float amount = properties.strength * edgeFade * distanceFade * mix(0.25, 1.0, grazing);
                        vec3 color = texture(sceneColor, hitUv).rgb;
                        FragColor = vec4(color, amount);
                    }
                "#,
        )
    ]
)
