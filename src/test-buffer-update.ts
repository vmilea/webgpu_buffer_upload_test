import { StagingBuffer, StagingBufferRing } from './staging';
import instancedShader from './shaders/instanced.wgsl';
import instancedSSBOShader from './shaders/instanced-ssbo.wgsl';
import naiveShader from './shaders/naive.wgsl';
import { assert, unreachable } from './utils/assert';
import { writeBuffer } from './utils/gpu';
import { roundUp } from './utils/math'
import { Mat4, mat4 } from 'wgpu-matrix';
import { Pane } from 'tweakpane';

const UNIFORM_BLOCK_SIZE = 256;

const SCENE_UNIFORM_SIZE = 64;
const MODEL_UNIFORM_SIZE = 64;
const VERTEX_SIZE = 24;
const INSTANCE_SIZE = 80;

const STAGING_RING_MAX_BUFFERS = 4;
const MAX_SQUARES = 250000;

const DEFAULT_BUFFER_SIZE = 128 * 1024 * 1024;

const MESH_COLORS = [
    [1, 0, 0, 1], // red 
    [0, 1, 0, 1], // green
    [0, 0, 1, 1], // blue
    [1, 1, 1, 1], // white
];

class Square {
    constructor(
        public x: number,
        public y: number,
        public speed: number,
        public scale: number,
        public meshIndex: number,
    ) { }

    getModelTransform(m: Mat4) {
        m[0] = this.scale; m[1] = 0; m[2] = 0; m[3] = 0;
        m[4] = 0; m[5] = this.scale; m[6] = 0; m[7] = 0;
        m[8] = 0; m[9] = 0; m[10] = 1; m[11] = 0;
        m[12] = this.x; m[13] = this.y; m[14] = 0; m[15] = 1;
    }

    writeVertices(dst: Float32Array, offset: number) {
        const color = MESH_COLORS[this.meshIndex];
        const vertices = [
            this.x - this.scale, this.y + this.scale, color[0], color[1], color[2], color[3],
            this.x - this.scale, this.y - this.scale, color[0], color[1], color[2], color[3],
            this.x + this.scale, this.y + this.scale, color[0], color[1], color[2], color[3],

            this.x + this.scale, this.y + this.scale, color[0], color[1], color[2], color[3],
            this.x - this.scale, this.y - this.scale, color[0], color[1], color[2], color[3],
            this.x + this.scale, this.y - this.scale, color[0], color[1], color[2], color[3],
        ];
        dst.set(vertices, offset);
    }
}

function createSquareMesh(device: GPUDevice, color: number[]): GPUBuffer {
    const vertices = [
        -1, 1, color[0], color[1], color[2], color[3],
        -1, -1, color[0], color[1], color[2], color[3],
        1, 1, color[0], color[1], color[2], color[3],

        1, 1, color[0], color[1], color[2], color[3],
        -1, -1, color[0], color[1], color[2], color[3],
        1, -1, color[0], color[1], color[2], color[3],
    ];

    const buffer = device.createBuffer({
        label: 'squareMesh',
        size: vertices.length * 4,
        usage: GPUBufferUsage.VERTEX,
        mappedAtCreation: true,
    });
    new Float32Array(buffer.getMappedRange()).set(vertices);
    buffer.unmap();
    return buffer;
}

function createNaiveRenderPipeline(device: GPUDevice, colorTargetFormat: GPUTextureFormat): [GPURenderPipeline, GPUBindGroupLayout] {
    const bindGroupLayoutEntries: GPUBindGroupLayoutEntry[] = [
        { binding: 0, visibility: GPUShaderStage.VERTEX, buffer: { type: 'uniform', hasDynamicOffset: true, minBindingSize: SCENE_UNIFORM_SIZE } },
        { binding: 1, visibility: GPUShaderStage.VERTEX, buffer: { type: 'uniform', hasDynamicOffset: true, minBindingSize: MODEL_UNIFORM_SIZE } },
    ];
    const bindGroupLayout = device.createBindGroupLayout({
        label: 'naiveBindGroupLayout',
        entries: bindGroupLayoutEntries,
    });
    const pipelineLayout = device.createPipelineLayout({
        label: 'naivePipelineLayout',
        bindGroupLayouts: [bindGroupLayout]
    });

    const shaderModule = device.createShaderModule({
        label: 'naiveShaderModule',
        code: naiveShader
    });
    const vertexAttributes: GPUVertexAttribute[] = [
        { format: 'float32x2', offset: 0, shaderLocation: 0 }, // position
        { format: 'float32x4', offset: 8, shaderLocation: 1 }, // color
    ];
    const vertexBufferLayouts: GPUVertexBufferLayout[] = [
        { arrayStride: VERTEX_SIZE, stepMode: 'vertex', attributes: vertexAttributes },
    ]
    const pipeline = device.createRenderPipeline({
        label: 'naivePipeline',
        layout: pipelineLayout,
        vertex: { module: shaderModule, buffers: vertexBufferLayouts },
        primitive: { topology: 'triangle-list' },
        fragment: { module: shaderModule, targets: [{ format: colorTargetFormat }] },
    });
    return [pipeline, bindGroupLayout];
}

function createInstancedRenderPipeline(device: GPUDevice, colorTargetFormat: GPUTextureFormat): [GPURenderPipeline, GPUBindGroupLayout] {
    const bindGroupLayoutEntries: GPUBindGroupLayoutEntry[] = [
        { binding: 0, visibility: GPUShaderStage.VERTEX, buffer: { type: 'uniform', hasDynamicOffset: true, minBindingSize: SCENE_UNIFORM_SIZE } },
    ];
    const bindGroupLayout = device.createBindGroupLayout({
        label: 'instancedBindGroupLayout',
        entries: bindGroupLayoutEntries,
    });
    const pipelineLayout = device.createPipelineLayout({
        label: 'instancedPipelineLayout',
        bindGroupLayouts: [bindGroupLayout]
    });

    const shaderModule = device.createShaderModule({
        label: 'instancedShaderModule',
        code: instancedShader
    });
    const vertexAttributes: GPUVertexAttribute[] = [
        { format: 'float32x2', offset: 0, shaderLocation: 0 }, // position
        { format: 'float32x4', offset: 8, shaderLocation: 1 }, // color
    ];
    const instanceAttributes: GPUVertexAttribute[] = [
        { format: 'float32x4', offset: 0, shaderLocation: 2 },  // instanceTransformCol0
        { format: 'float32x4', offset: 16, shaderLocation: 3 }, // instanceTransformCol1
        { format: 'float32x4', offset: 32, shaderLocation: 4 }, // instanceTransformCol2
        { format: 'float32x4', offset: 48, shaderLocation: 5 }, // instanceTransformCol3
        { format: 'float32x4', offset: 64, shaderLocation: 6 }, // instanceColor
    ];
    const vertexBufferLayouts: GPUVertexBufferLayout[] = [
        { arrayStride: VERTEX_SIZE, stepMode: 'vertex', attributes: vertexAttributes },
        { arrayStride: INSTANCE_SIZE, stepMode: 'instance', attributes: instanceAttributes },
    ]
    const pipeline = device.createRenderPipeline({
        label: 'instancedPipeline',
        layout: pipelineLayout,
        vertex: { module: shaderModule, buffers: vertexBufferLayouts },
        primitive: { topology: 'triangle-list' },
        fragment: { module: shaderModule, targets: [{ format: colorTargetFormat }] },
    });
    return [pipeline, bindGroupLayout];
}

function createInstancedSSBORenderPipeline(device: GPUDevice, colorTargetFormat: GPUTextureFormat): [GPURenderPipeline, GPUBindGroupLayout] {
    const bindGroupLayoutEntries: GPUBindGroupLayoutEntry[] = [
        { binding: 0, visibility: GPUShaderStage.VERTEX, buffer: { type: 'uniform', hasDynamicOffset: false, minBindingSize: SCENE_UNIFORM_SIZE } },
        { binding: 1, visibility: GPUShaderStage.VERTEX, buffer: { type: 'read-only-storage', hasDynamicOffset: false, minBindingSize: 0 } },
    ];
    const bindGroupLayout = device.createBindGroupLayout({
        label: 'instancedSSBOBindGroupLayout',
        entries: bindGroupLayoutEntries,
    });
    const pipelineLayout = device.createPipelineLayout({
        label: 'instancedSSBOPipelineLayout',
        bindGroupLayouts: [bindGroupLayout]
    });

    const shaderModule = device.createShaderModule({
        label: 'instancedSSBOShaderModule',
        code: instancedSSBOShader
    });
    const vertexAttributes: GPUVertexAttribute[] = [
        { format: 'float32x2', offset: 0, shaderLocation: 0 }, // position
        { format: 'float32x4', offset: 8, shaderLocation: 1 }, // color
    ];
    const vertexBufferLayouts: GPUVertexBufferLayout[] = [
        { arrayStride: VERTEX_SIZE, stepMode: 'vertex', attributes: vertexAttributes },
    ]
    const pipeline = device.createRenderPipeline({
        label: 'instancedSSBOPipeline',
        layout: pipelineLayout,
        vertex: { module: shaderModule, buffers: vertexBufferLayouts },
        primitive: { topology: 'triangle-list' },
        fragment: { module: shaderModule, targets: [{ format: colorTargetFormat }] },
    });
    return [pipeline, bindGroupLayout];
}

type DrawMode =
    | 'naive' // separate draw per item (static mesh, streams uniforms)
    | 'batchedVertices' // batched items (streams mesh vertices)
    | 'batchedInstances' // batched items (static mesh, streams instance data - vertex attributes)
    | 'batchedInstancesSSBO' // batched items (static mesh, streams instance data - SSBO)

type UpdateMode =
    | 'default' // writeBuffer()
    | 'map' // mapped memory, mapping the entire staging buffer
    | 'mapRequired' // mapped memory, mapping just the required range

interface Context {
    device: GPUDevice,
    canvasContext: GPUCanvasContext,
    surfaceWidth: number,
    surfaceHeight: number,
    surfaceFormat: GPUTextureFormat,

    frameCounter: number,
    fpsAccumulator: number,
    previousTime: DOMHighResTimeStamp,

    meshes: GPUBuffer[],

    // device GPUBuffer (just one, shouldn't cause pipeline stalls since it's only accessed from vertex shader)
    streamBuffer: GPUBuffer,
    // ring of host-visible GPUBuffers for staging
    stagingRing: StagingBufferRing,
    // host memory buffer as a fallback from ring buffers
    stagingArrayBuffer: ArrayBuffer,

    naivePipeline: GPURenderPipeline,
    naiveBindGroupLayout: GPUBindGroupLayout,
    instancedPipeline: GPURenderPipeline,
    instancedBindGroupLayout: GPUBindGroupLayout,
    instancedSSBOPipeline: GPURenderPipeline,
    instancedSSBOBindGroupLayout: GPUBindGroupLayout,

    renderBundle?: GPURenderBundle,

    // scene
    projectionTransform: Mat4,
    squares: Square[],
    
    // stats
    fps: number,
    uploadSize: number,

    // tweakables
    bufferSize: number,
    updateMode: UpdateMode,
    drawMode: DrawMode,
    squareCount: number,
    batchSize: number,
    useRenderBundle: boolean,
}

function adjustBufferSize(g: Context) {
    if (g.stagingArrayBuffer.byteLength === g.bufferSize) {
        return; // nothing to do
    }
    
    // destroying the buffers marks them for deallocation after all referencing GPU commands has completed;
    // since we can't easily wait for that before creating the new buffers, there will be a spike in memory
    // footprint
    g.streamBuffer.destroy();
    g.stagingRing.destroy();
    g.streamBuffer = g.device.createBuffer({ size: g.bufferSize, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.VERTEX | GPUBufferUsage.UNIFORM });
    g.stagingRing = new StagingBufferRing(g.device, g.bufferSize, STAGING_RING_MAX_BUFFERS);
    g.stagingArrayBuffer = new ArrayBuffer(g.bufferSize);
}

function tick(g: Context, dt: number) {
    for (const square of g.squares) {
        square.y -= dt * square.speed;
        if (square.y < 0) {
            square.y += g.surfaceHeight;
        }
    }
}

function render(g: Context) {
    if (g.surfaceWidth == 0 || g.surfaceHeight == 0) {
        return;
    }

    adjustBufferSize(g);

    let streamData: Float32Array;
    let uniformDataSize = 0;
    let vertexDataSize = 0;
    let stagingBuffer: StagingBuffer | undefined = undefined;
    let stagingSize = g.bufferSize;
    if (g.updateMode == 'default') {
        streamData = new Float32Array(g.stagingArrayBuffer);
    } else {
        if (g.updateMode == 'mapRequired') {
            // guesstimate the buffer space required in order to optimize mapping
            // (nontrivial apps will have a hard time doing this)
            if (g.drawMode == 'naive') {
                stagingSize = (1 + g.squareCount) * UNIFORM_BLOCK_SIZE;
            } else if (g.drawMode == 'batchedVertices') {
                stagingSize = (1 + 1) * UNIFORM_BLOCK_SIZE + g.squareCount * (6 * VERTEX_SIZE);
            } else if (g.drawMode == 'batchedInstances' || g.drawMode == 'batchedInstancesSSBO') {
                stagingSize = UNIFORM_BLOCK_SIZE + g.squareCount * INSTANCE_SIZE;
            } else {
                unreachable();
            }
            stagingSize = Math.max(UNIFORM_BLOCK_SIZE, roundUp(stagingSize, UNIFORM_BLOCK_SIZE));
        }
        assert(stagingSize <= g.bufferSize);

        stagingBuffer = g.stagingRing.next();
        if (stagingBuffer !== undefined && stagingSize <= stagingBuffer.mappingSize) {
            if (stagingSize == stagingBuffer.mappingSize)  {
                // console.log("mapping is the right size");
            } else {
                console.log("mapping is larger than required");
            }
            streamData = new Float32Array(stagingBuffer.buffer.getMappedRange(0, stagingSize));
        } else {
            if (stagingBuffer === undefined) {
                console.log("no staging buffer is available, fall back to writeBuffer()");
            } else {
                console.log("mapping is too small, fall back to writeBuffer()");
            }
            streamData = new Float32Array(g.stagingArrayBuffer);
        }
    }

    // write scene uniform
    streamData.set(g.projectionTransform, uniformDataSize / 4);
    uniformDataSize += UNIFORM_BLOCK_SIZE;

    const commandEncoder = g.device.createCommandEncoder();
    const colorAttachment: GPURenderPassColorAttachment = {
        view: g.canvasContext.getCurrentTexture().createView(),
        clearValue: [0, 0, 0, 1],
        loadOp: 'clear',
        storeOp: 'store',
    };
    const renderEncoder = commandEncoder.beginRenderPass({
        label: 'squaresRenderPass',
        colorAttachments: [colorAttachment],
    });

    let draw: (renderEncoder: GPUBindingCommandsMixin & GPURenderCommandsMixin) => void;

    if (g.drawMode == 'naive') {
        assert(uniformDataSize + g.squareCount * UNIFORM_BLOCK_SIZE <= g.bufferSize);

        const modelTransform = mat4.create();
        for (let i = 0; i < g.squareCount; i++) {
            const square = g.squares[i];
            square.getModelTransform(modelTransform);
            // write model uniform
            streamData.set(modelTransform, uniformDataSize / 4);
            uniformDataSize += UNIFORM_BLOCK_SIZE;
        }

        draw = (renderEncoder: GPUBindingCommandsMixin & GPURenderCommandsMixin) => {
            renderEncoder.setPipeline(g.naivePipeline);

            const bindGroup = g.device.createBindGroup({
                label: 'naiveBindGroup',
                layout: g.naiveBindGroupLayout,
                entries: [
                    { binding: 0, resource: { buffer: g.streamBuffer, offset: 0, size: SCENE_UNIFORM_SIZE } },
                    { binding: 1, resource: { buffer: g.streamBuffer, offset: 0, size: MODEL_UNIFORM_SIZE } },
                ],
            });

            let dynamicOffset = UNIFORM_BLOCK_SIZE;
            for (let i = 0; i < g.squareCount; i++) {
                const square = g.squares[i];
                // draw item
                const dynamicOffsets = [0, dynamicOffset];
                renderEncoder.setBindGroup(0, bindGroup, dynamicOffsets);
                renderEncoder.setVertexBuffer(0, g.meshes[square.meshIndex]);
                renderEncoder.draw(6);
                dynamicOffset += UNIFORM_BLOCK_SIZE;
            }
        }
    } else if (g.drawMode == 'batchedVertices') {
        assert(g.squareCount * 6 * VERTEX_SIZE <= g.bufferSize);

        // write model uniform
        const modelTransform = mat4.identity();
        streamData.set(modelTransform, uniformDataSize / 4);
        uniformDataSize += UNIFORM_BLOCK_SIZE;

        // write vertices
        for (let i = 0; i < g.squareCount; i++) {
            const square = g.squares[i];
            square.writeVertices(streamData, (uniformDataSize + vertexDataSize) / 4);
            vertexDataSize += 6 * VERTEX_SIZE;
        }

        draw = (renderEncoder: GPUBindingCommandsMixin & GPURenderCommandsMixin) => {
            renderEncoder.setPipeline(g.naivePipeline);

            const bindGroup = g.device.createBindGroup({
                label: 'naiveBindGroup',
                layout: g.naiveBindGroupLayout,
                entries: [
                    { binding: 0, resource: { buffer: g.streamBuffer, offset: 0, size: SCENE_UNIFORM_SIZE } },
                    { binding: 1, resource: { buffer: g.streamBuffer, offset: 0, size: MODEL_UNIFORM_SIZE } },
                ],
            });
            const dynamicOffsets = [0, UNIFORM_BLOCK_SIZE];
            renderEncoder.setBindGroup(0, bindGroup, dynamicOffsets);
            renderEncoder.setVertexBuffer(0, g.streamBuffer, uniformDataSize);

            // draw in batches
            for (let i = 0; i < g.squareCount; i += g.batchSize) {
                const count = Math.min(g.batchSize, g.squareCount - i);
                renderEncoder.draw(count * 6, 1, i * 6, 0);
            }
        }
    } else if (g.drawMode == 'batchedInstances' || g.drawMode == 'batchedInstancesSSBO') {
        assert(g.squareCount * INSTANCE_SIZE <= g.bufferSize);

        // write instance data
        const modelTransform = mat4.create();
        for (let i = 0; i < g.squareCount; i++) {
            const square = g.squares[i];
            square.getModelTransform(modelTransform);
            const floatOffset = (uniformDataSize + vertexDataSize) / 4;
            streamData.set(modelTransform, floatOffset);
            streamData.set(MESH_COLORS[square.meshIndex], floatOffset + 16);
            vertexDataSize += INSTANCE_SIZE;
        }

        if (g.drawMode == 'batchedInstances') {
            draw = (renderEncoder: GPUBindingCommandsMixin & GPURenderCommandsMixin) => {
                renderEncoder.setPipeline(g.instancedPipeline);

                const bindGroup = g.device.createBindGroup({
                    label: 'instancedBindGroup',
                    layout: g.instancedBindGroupLayout,
                    entries: [
                        { binding: 0, resource: { buffer: g.streamBuffer, offset: 0, size: SCENE_UNIFORM_SIZE } },
                    ],
                });
                const dynamicOffsets = [0];
                renderEncoder.setBindGroup(0, bindGroup, dynamicOffsets);
                // will draw instances of the white mesh; transform and color are instance attributes
                renderEncoder.setVertexBuffer(0, g.meshes[3]);
                renderEncoder.setVertexBuffer(1, g.streamBuffer, uniformDataSize);

                // draw in batches
                for (let i = 0; i < g.squareCount; i += g.batchSize) {
                    const count = Math.min(g.batchSize, g.squareCount - i);
                    renderEncoder.draw(6, count, 0, i);
                }
            }
        } else {
            draw = (renderEncoder: GPUBindingCommandsMixin & GPURenderCommandsMixin) => {
                renderEncoder.setPipeline(g.instancedSSBOPipeline);
    
                const bindGroup = g.device.createBindGroup({
                    label: 'instancedSSBOBindGroup',
                    layout: g.instancedSSBOBindGroupLayout,
                    entries: [
                        { binding: 0, resource: { buffer: g.streamBuffer, offset: 0, size: SCENE_UNIFORM_SIZE } },
                        { binding: 1, resource: { buffer: g.streamBuffer, offset: uniformDataSize, size: g.squareCount * INSTANCE_SIZE } },
                    ],
                });
                renderEncoder.setBindGroup(0, bindGroup);
                // will draw instances of the white mesh; transform and color are read from SSBO
                renderEncoder.setVertexBuffer(0, g.meshes[3]);
    
                // draw in batches
                for (let i = 0; i < g.squareCount; i += g.batchSize) {
                    const count = Math.min(g.batchSize, g.squareCount - i);
                    renderEncoder.draw(6, count, 0, i);
                }
            }
        }
    } else {
        unreachable();
    }

    // done preparing frame data
    g.uploadSize = uniformDataSize + vertexDataSize;
    if (stagingBuffer !== undefined) {
        stagingBuffer.buffer.unmap();
    }

    if (g.useRenderBundle) {
        if (g.renderBundle == undefined) {
            const bundleEncoder = g.device.createRenderBundleEncoder({ colorFormats: [g.surfaceFormat] });
            draw(bundleEncoder);
            g.renderBundle = bundleEncoder.finish();
        }
        renderEncoder.executeBundles([g.renderBundle]);
    } else {
        draw(renderEncoder);
    }
    renderEncoder.end();

    const queue = g.device.queue;
    // copy staging data before rendering
    if (streamData.buffer === g.stagingArrayBuffer) {
        writeBuffer(queue, g.streamBuffer, g.stagingArrayBuffer, g.uploadSize);
        queue.submit([commandEncoder.finish()]);
    } else {
        assert(stagingBuffer !== undefined);
        const copyCommandEncoder = g.device.createCommandEncoder();
        copyCommandEncoder.copyBufferToBuffer(stagingBuffer.buffer, 0, g.streamBuffer, 0, g.uploadSize);
        queue.submit([copyCommandEncoder.finish(), commandEncoder.finish()]);
    }

    // staging buffer will be reclaimed after copying on the queue timeline
    if (stagingBuffer !== undefined) {
        g.stagingRing.remap(stagingBuffer, stagingSize);
    }
}

function tweakpane(g: Context) {
    const pane = new Pane({ expanded: true });

    pane.addBinding(g, 'fps', {
        label: 'FPS',
        readonly: true,
        format: (value) => Math.round(value).toString(),
    });

    pane.addBinding(g, 'fps', {
        label: 'Frame time (ms)',
        readonly: true,
        format: (value) => (1000 / value).toFixed(2),
    });

    pane.addBinding(g, 'uploadSize', {
        label: 'Upload size (MB)',
        readonly: true,
        format: (value) => Math.ceil(value / (1024 * 1024)).toString(),
    });

    const controlsFolder = pane.addFolder({
        title: 'Controls',
    });

    const bindingState = { value: g.bufferSize / (1024 * 1024) };
    controlsFolder.addBinding(bindingState, 'value', {
        label: 'Buffer size (MB)',
        min: Math.ceil(MAX_SQUARES * 256 / (1024 * 1024)),
        max: 256,
        step: 1,
    }).on('change', (ev) => {
        if (ev.last) {
            g.bufferSize = bindingState.value * (1024 * 1024);
            g.renderBundle = undefined;
        }
    });
    
    controlsFolder.addBinding(g, 'updateMode', {
        label: 'Update mode',
        options: { 'Default': 'default', 'Map (whole buffer)': 'map', 'Map (min size)': 'mapRequired' },
    });
    
    controlsFolder.addBinding(g, 'drawMode', {
        label: 'Draw mode',
        options: {
            'Naive': 'naive',
            'Batched vertices': 'batchedVertices',
            'Batched instances': 'batchedInstances',
            'Batched instances (SSBO)': 'batchedInstancesSSBO',
        },
    }).on('change', () => {
        g.renderBundle = undefined;
    });
    
    controlsFolder.addBinding(g, 'squareCount', {
        label: 'Squares',
        min: 1,
        max: MAX_SQUARES,
        step: 1,
    }).on('change', () => {
        g.renderBundle = undefined;
    });
    
    controlsFolder.addBinding(g, 'batchSize', {
        label: 'Batch size',
        min: 1,
        max: MAX_SQUARES,
        step: 1,
    }).on('change', () => {
        g.renderBundle = undefined;
    });
    
    controlsFolder.addBinding(g, 'useRenderBundle', {
        label: 'Use render bundle',
    }).on('change', () => {
        g.renderBundle = undefined;
    });
}

export default function testBufferUpdate(canvasContext: GPUCanvasContext, device: GPUDevice): void {
    const surfaceFormat = navigator.gpu.getPreferredCanvasFormat();
    canvasContext.configure({
        device: device,
        format: surfaceFormat,
        alphaMode: 'opaque',
    });
    const canvas = canvasContext.canvas as HTMLCanvasElement;

    const [naivePipeline, naiveBindGroupLayout] = createNaiveRenderPipeline(device, surfaceFormat);
    const [instancedPipeline, instancedBindGroupLayout] = createInstancedRenderPipeline(device, surfaceFormat);
    const [instancedSSBOPipeline, instancedSSBOBindGroupLayout] = createInstancedSSBORenderPipeline(device, surfaceFormat);

    const squares = Array<Square>(MAX_SQUARES);
    for (let i = 0; i < MAX_SQUARES; i++) {
        const speed = 50 + Math.random() * 150;
        const scale = 5 + Math.random() * 5;
        const meshIndex = Math.floor(Math.random() * 3);
        assert(meshIndex < 3);
        squares[i] = new Square(0, 0, speed, scale, meshIndex);
    }

    const g: Context = {
        device: device,
        canvasContext: canvasContext,
        surfaceWidth: 0,
        surfaceHeight: 0,
        surfaceFormat: surfaceFormat,
        frameCounter: 0,
        fpsAccumulator: 0,
        previousTime: 0,
        meshes: MESH_COLORS.map(color => createSquareMesh(device, color)),
        streamBuffer: device.createBuffer({ size: DEFAULT_BUFFER_SIZE, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.VERTEX | GPUBufferUsage.UNIFORM | GPUBufferUsage.STORAGE }),
        stagingRing: new StagingBufferRing(device, DEFAULT_BUFFER_SIZE, STAGING_RING_MAX_BUFFERS),
        stagingArrayBuffer: new ArrayBuffer(DEFAULT_BUFFER_SIZE),
        naivePipeline: naivePipeline,
        naiveBindGroupLayout: naiveBindGroupLayout,
        instancedPipeline: instancedPipeline,
        instancedBindGroupLayout: instancedBindGroupLayout,
        instancedSSBOPipeline: instancedSSBOPipeline,
        instancedSSBOBindGroupLayout: instancedSSBOBindGroupLayout,
        renderBundle: undefined,
        projectionTransform: mat4.create(),
        squares: squares,
        fps: 0,
        uploadSize: 0,
        bufferSize: DEFAULT_BUFFER_SIZE,
        updateMode: 'mapRequired',
        drawMode: 'batchedInstances',
        squareCount: MAX_SQUARES,
        batchSize: 1,
        useRenderBundle: true,
    }

    tweakpane(g);

    function frame(time: DOMHighResTimeStamp) {
        const previousSeconds = Math.floor(g.previousTime / 1000);
        const seconds = Math.floor(time / 1000);
        if (previousSeconds != seconds && 0 < g.fpsAccumulator) {
            g.fps = g.fpsAccumulator;
            g.fpsAccumulator = 0;
        }

        const dt = (g.previousTime == 0 ? 0 : ((time - g.previousTime) / 1000));
        tick(g, dt);
        render(g);

        g.frameCounter++;
        g.fpsAccumulator++;
        g.previousTime = time;
        requestAnimationFrame(frame);
    }

    const observer = new ResizeObserver(() => {
        const width = canvas.clientWidth;
        const height = canvas.clientHeight;
        g.surfaceWidth = canvas.width = width;
        g.surfaceHeight = canvas.height = height;

        mat4.ortho(0, width, 0, height, 0, -1, g.projectionTransform);

        for (const square of g.squares) {
            square.x = Math.random() * width;
            square.y = Math.random() * height;
        }

        requestAnimationFrame(frame);
    });
    observer.observe(canvas);
}
