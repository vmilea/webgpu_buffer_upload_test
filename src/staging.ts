import { assert } from "./utils/assert";

export class StagingBuffer {
    readonly buffer: GPUBuffer;
    mappingSize: number;

    constructor(device: GPUDevice, size: number) {
        console.log(`new StagingBuffer - size: ${size}`);

        this.buffer = device.createBuffer({
            label: 'stagingBuffer',
            size: size,
            usage: GPUBufferUsage.MAP_WRITE | GPUBufferUsage.COPY_SRC,
            mappedAtCreation: true,
        });
        this.mappingSize = size;
    }
}

export class StagingBufferRing {
    readonly #mappedBuffers = new Array<StagingBuffer>();
    readonly #pendingBuffers = new Set<StagingBuffer>([]);
    #destroyed = false;

    constructor(
        public device: GPUDevice,
        public bufferSize: number,
        public maxBuffers: number) {
    }

    next() {
        assert(!this.#destroyed);

        if (0 < this.#mappedBuffers.length) {
            return this.#mappedBuffers.shift();
        }
        if (this.#pendingBuffers.size < this.maxBuffers) {
            return new StagingBuffer(this.device, this.bufferSize);
        }
        return undefined;
    }

    remap(stagingBuffer: StagingBuffer, mappingSize: number) {
        assert(!this.#destroyed);
        
        this.#pendingBuffers.add(stagingBuffer);
        
        stagingBuffer.buffer.mapAsync(GPUMapMode.WRITE, 0, mappingSize).then(() => {
            this.#pendingBuffers.delete(stagingBuffer);
            if (this.#destroyed) {
                stagingBuffer.buffer.destroy();
            } else {
                stagingBuffer.mappingSize = mappingSize;
                this.#mappedBuffers.push(stagingBuffer);
                // console.log(`mapAsync() finished - available: ${this.#mappedBuffers.length} / ${this.#pendingBuffers.size + this.#mappedBuffers.length}`);
            }
        }, (reason) => {
            this.#pendingBuffers.delete(stagingBuffer);
            console.error(`mapAsync() failed - ${reason}`);
        });
    }

    destroy() {
        assert(!this.#destroyed);

        for (const buffer of this.#mappedBuffers) {
            buffer.buffer.destroy();
        }
        this.#mappedBuffers.length = 0;
        this.#destroyed = true;
    }
}
