struct SceneUniform {
    viewProjectionTransform: mat4x4f,
}

struct Uniform {
    modelTransform: mat4x4f,
}

@group(0) @binding(0) var<uniform> uScene: SceneUniform;
@group(0) @binding(1) var<uniform> u: Uniform;

struct VertexInput {
    @location(0) position: vec2f,
    @location(1) color: vec4f, // premultiplied
};

struct VertexOutput {
    @builtin(position) position: vec4f,
    @location(0) color: vec4f, // premultiplied
}

@vertex fn vertexMain(in: VertexInput) -> VertexOutput {
    let position = uScene.viewProjectionTransform * (u.modelTransform * vec4f(in.position, 0, 1));
    return VertexOutput(position, in.color);
}

@fragment fn fragmentMain(in: VertexOutput) -> @location(0) vec4f {
    return in.color;
}
