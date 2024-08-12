struct SceneUniform {
    viewProjectionTransform: mat4x4f,
}

@group(0) @binding(0) var<uniform> uScene: SceneUniform;

struct VertexInput {
    @location(0) position: vec2f,
    @location(1) color: vec4f, // premultiplied
    @location(2) instanceTransformCol0: vec4f,
    @location(3) instanceTransformCol1: vec4f,
    @location(4) instanceTransformCol2: vec4f,
    @location(5) instanceTransformCol3: vec4f,
    @location(6) instanceColor: vec4f, // premultiplied
};

struct VertexOutput {
    @builtin(position) position: vec4f,
    @location(0) color: vec4f, // premultiplied
}

@vertex fn vertexMain(in: VertexInput) -> VertexOutput {
    let instanceTransform = mat4x4f(in.instanceTransformCol0,
                                    in.instanceTransformCol1,
                                    in.instanceTransformCol2,
                                    in.instanceTransformCol3);
    let position = uScene.viewProjectionTransform * (instanceTransform * vec4f(in.position, 0, 1));
    let color = in.color * in.instanceColor;
    return VertexOutput(position, color);
}

@fragment fn fragmentMain(in: VertexOutput) -> @location(0) vec4f {
    return in.color;
}
