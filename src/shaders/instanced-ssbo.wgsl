struct SceneUniform {
    viewProjectionTransform: mat4x4f,
}

struct Instance {
    transform : mat4x4f,
    color : vec4f, // premultiplied
}

@group(0) @binding(0) var<uniform> uScene: SceneUniform;
@group(0) @binding(1) var<storage, read> gInstances: array<Instance>;

struct VertexInput {
    @location(0) position: vec2f,
    @location(1) color: vec4f, // premultiplied
};

struct VertexOutput {
    @builtin(position) position: vec4f,
    @location(0) color: vec4f, // premultiplied
}

@vertex fn vertexMain(@builtin(instance_index) instanceIndex: u32, in: VertexInput) -> VertexOutput {
    let instance = gInstances[instanceIndex];
    let position = uScene.viewProjectionTransform * (instance.transform * vec4f(in.position, 0, 1));
    let color = in.color * instance.color;
    return VertexOutput(position, color);
}

@fragment fn fragmentMain(in: VertexOutput) -> @location(0) vec4f {
    return in.color;
}
