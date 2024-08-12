import './style.css'
import testBufferUpdate from './test-buffer-update';
import { assert } from './utils/assert';

(async () => {
    if (navigator.gpu === undefined) {
        const h = document.querySelector('#title') as HTMLElement;
        h.innerText = 'WebGPU is not supported in this browser.';
        return;
    }
    const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
    if (adapter === null) {
        const h = document.querySelector('#title') as HTMLElement;
        h.innerText = 'No adapter is available for WebGPU.';
        return;
    }
    const device = await adapter.requestDevice();

    const canvas = document.querySelector<HTMLCanvasElement>('#webgpu-canvas');
    assert(canvas !== null);
    const context = canvas.getContext('webgpu')!;

    testBufferUpdate(context, device);
})();