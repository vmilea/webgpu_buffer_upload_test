export function writeBuffer(queue: GPUQueue, dst: GPUBuffer, src: ArrayBuffer, size: number) {
    // large writes (16MiB+) are slow on NVIDIA, split into chunks
    const CHUNK_SIZE = 4 * 1024 * 1024;

    let offset = 0;
    while (offset < size) {
        const chunkSize = Math.min(size - offset, CHUNK_SIZE);
        queue.writeBuffer(dst, offset, src, offset, chunkSize);
        offset += chunkSize;
    }
}