// Inferencia de Red Neuronal con WebGPU
import { neuralNetWeights, createInputVector, foodCategories, softmax } from './neuralnet.js';

export class NeuralNetGPU {
  constructor(device) {
    this.device = device;
    this.initialized = false;
  }

  async initialize() {
    if (!this.device) return false;

    try {
      // Crear buffers para los pesos de la red neuronal
      this.createBuffers();
      
      // Crear pipelines de compute shaders
      await this.createPipelines();
      
      this.initialized = true;
      console.log('🧠 Red neuronal GPU inicializada');
      return true;
    } catch (error) {
      console.error('Error inicializando red neuronal GPU:', error);
      return false;
    }
  }

  createBuffers() {
    const weights = neuralNetWeights;

    // Buffer para pesos de capa 1 (8 inputs x 16 hidden)
    this.layer1WeightsBuffer = this.device.createBuffer({
      size: weights.layer1.weights.byteLength,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      mappedAtCreation: true
    });
    new Float32Array(this.layer1WeightsBuffer.getMappedRange()).set(weights.layer1.weights);
    this.layer1WeightsBuffer.unmap();

    // Buffer para biases de capa 1
    this.layer1BiasesBuffer = this.device.createBuffer({
      size: weights.layer1.biases.byteLength,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      mappedAtCreation: true
    });
    new Float32Array(this.layer1BiasesBuffer.getMappedRange()).set(weights.layer1.biases);
    this.layer1BiasesBuffer.unmap();

    // Buffer para pesos de capa 2 (16 hidden x 12 outputs)
    this.layer2WeightsBuffer = this.device.createBuffer({
      size: weights.layer2.weights.byteLength,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      mappedAtCreation: true
    });
    new Float32Array(this.layer2WeightsBuffer.getMappedRange()).set(weights.layer2.weights);
    this.layer2WeightsBuffer.unmap();

    // Buffer para biases de capa 2
    this.layer2BiasesBuffer = this.device.createBuffer({
      size: weights.layer2.biases.byteLength,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      mappedAtCreation: true
    });
    new Float32Array(this.layer2BiasesBuffer.getMappedRange()).set(weights.layer2.biases);
    this.layer2BiasesBuffer.unmap();
  }

  async createPipelines() {
    // Shader para la capa 1: Input -> Hidden con ReLU
    const layer1ShaderCode = `
      @group(0) @binding(0) var<storage, read> input: array<f32, 8>;
      @group(0) @binding(1) var<storage, read> weights: array<f32, 128>; // 8x16
      @group(0) @binding(2) var<storage, read> biases: array<f32, 16>;
      @group(0) @binding(3) var<storage, read_write> output: array<f32, 16>;

      @compute @workgroup_size(16)
      fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
        let neuron_idx = global_id.x;
        if (neuron_idx >= 16u) {
          return;
        }

        var sum = 0.0;
        for (var i = 0u; i < 8u; i = i + 1u) {
          let weight_idx = neuron_idx * 8u + i;
          sum = sum + input[i] * weights[weight_idx];
        }
        sum = sum + biases[neuron_idx];
        
        // ReLU activation
        output[neuron_idx] = max(0.0, sum);
      }
    `;

    // Shader para la capa 2: Hidden -> Output con ReLU
    const layer2ShaderCode = `
      @group(0) @binding(0) var<storage, read> input: array<f32, 16>;
      @group(0) @binding(1) var<storage, read> weights: array<f32, 192>; // 16x12
      @group(0) @binding(2) var<storage, read> biases: array<f32, 12>;
      @group(0) @binding(3) var<storage, read_write> output: array<f32, 12>;

      @compute @workgroup_size(12)
      fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
        let neuron_idx = global_id.x;
        if (neuron_idx >= 12u) {
          return;
        }

        var sum = 0.0;
        for (var i = 0u; i < 16u; i = i + 1u) {
          let weight_idx = neuron_idx * 16u + i;
          sum = sum + input[i] * weights[weight_idx];
        }
        sum = sum + biases[neuron_idx];
        
        // ReLU activation
        output[neuron_idx] = max(0.0, sum);
      }
    `;

    // Crear módulos de shader
    const layer1Module = this.device.createShaderModule({ code: layer1ShaderCode });
    const layer2Module = this.device.createShaderModule({ code: layer2ShaderCode });

    // Crear layout para capa 1
    const layer1BindGroupLayout = this.device.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
        { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
        { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
        { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }
      ]
    });

    // Crear layout para capa 2
    const layer2BindGroupLayout = this.device.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
        { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
        { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
        { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }
      ]
    });

    // Crear pipelines
    this.layer1Pipeline = this.device.createComputePipeline({
      layout: this.device.createPipelineLayout({ bindGroupLayouts: [layer1BindGroupLayout] }),
      compute: { module: layer1Module, entryPoint: 'main' }
    });

    this.layer2Pipeline = this.device.createComputePipeline({
      layout: this.device.createPipelineLayout({ bindGroupLayouts: [layer2BindGroupLayout] }),
      compute: { module: layer2Module, entryPoint: 'main' }
    });

    // Guardar layouts para crear bind groups después
    this.layer1BindGroupLayout = layer1BindGroupLayout;
    this.layer2BindGroupLayout = layer2BindGroupLayout;
  }

  async predict(grape, year) {
    if (!this.initialized) {
      return this.fallbackPredict(grape, year);
    }

    try {
      // Crear vector de entrada
      const inputVector = createInputVector(grape, year);

      // Crear buffer de entrada
      const inputBuffer = this.device.createBuffer({
        size: inputVector.byteLength,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        mappedAtCreation: true
      });
      new Float32Array(inputBuffer.getMappedRange()).set(inputVector);
      inputBuffer.unmap();

      // Crear buffer para salida de capa 1 (hidden layer)
      const hiddenBuffer = this.device.createBuffer({
        size: Float32Array.BYTES_PER_ELEMENT * 16,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC
      });

      // Crear buffer para salida final
      const outputBuffer = this.device.createBuffer({
        size: Float32Array.BYTES_PER_ELEMENT * 12,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC
      });

      // Crear bind group para capa 1
      const layer1BindGroup = this.device.createBindGroup({
        layout: this.layer1BindGroupLayout,
        entries: [
          { binding: 0, resource: { buffer: inputBuffer } },
          { binding: 1, resource: { buffer: this.layer1WeightsBuffer } },
          { binding: 2, resource: { buffer: this.layer1BiasesBuffer } },
          { binding: 3, resource: { buffer: hiddenBuffer } }
        ]
      });

      // Crear bind group para capa 2
      const layer2BindGroup = this.device.createBindGroup({
        layout: this.layer2BindGroupLayout,
        entries: [
          { binding: 0, resource: { buffer: hiddenBuffer } },
          { binding: 1, resource: { buffer: this.layer2WeightsBuffer } },
          { binding: 2, resource: { buffer: this.layer2BiasesBuffer } },
          { binding: 3, resource: { buffer: outputBuffer } }
        ]
      });

      // Ejecutar inferencia
      const commandEncoder = this.device.createCommandEncoder();
      
      // Capa 1
      const pass1 = commandEncoder.beginComputePass();
      pass1.setPipeline(this.layer1Pipeline);
      pass1.setBindGroup(0, layer1BindGroup);
      pass1.dispatchWorkgroups(1);
      pass1.end();

      // Capa 2
      const pass2 = commandEncoder.beginComputePass();
      pass2.setPipeline(this.layer2Pipeline);
      pass2.setBindGroup(0, layer2BindGroup);
      pass2.dispatchWorkgroups(1);
      pass2.end();

      // Copiar resultados a buffer legible
      const readBuffer = this.device.createBuffer({
        size: Float32Array.BYTES_PER_ELEMENT * 12,
        usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ
      });

      commandEncoder.copyBufferToBuffer(
        outputBuffer, 0,
        readBuffer, 0,
        Float32Array.BYTES_PER_ELEMENT * 12
      );

      this.device.queue.submit([commandEncoder.finish()]);

      // Leer resultados
      await readBuffer.mapAsync(GPUMapMode.READ);
      const resultData = new Float32Array(readBuffer.getMappedRange()).slice();
      readBuffer.unmap();

      // Aplicar softmax para obtener probabilidades
      const probabilities = softmax(Array.from(resultData));

      // Obtener top 4 categorías con mayor probabilidad
      const predictions = probabilities
        .map((prob, idx) => ({ category: foodCategories[idx], score: prob }))
        .sort((a, b) => b.score - a.score)
        .slice(0, 4);

      return {
        predictions,
        usedGPU: true
      };

    } catch (error) {
      console.error('Error en predicción GPU:', error);
      return this.fallbackPredict(grape, year);
    }
  }

  // Predicción sin GPU (fallback)
  fallbackPredict(grape, year) {
    const inputVector = createInputVector(grape, year);
    const weights = neuralNetWeights;

    // Capa 1: Input -> Hidden con ReLU
    const hidden = [];
    for (let i = 0; i < 16; i++) {
      let sum = weights.layer1.biases[i];
      for (let j = 0; j < 8; j++) {
        sum += inputVector[j] * weights.layer1.weights[i * 8 + j];
      }
      hidden.push(Math.max(0, sum)); // ReLU
    }

    // Capa 2: Hidden -> Output con ReLU
    const output = [];
    for (let i = 0; i < 12; i++) {
      let sum = weights.layer2.biases[i];
      for (let j = 0; j < 16; j++) {
        sum += hidden[j] * weights.layer2.weights[i * 16 + j];
      }
      output.push(Math.max(0, sum)); // ReLU
    }

    // Aplicar softmax
    const probabilities = softmax(output);

    // Obtener top 4 categorías
    const predictions = probabilities
      .map((prob, idx) => ({ category: foodCategories[idx], score: prob }))
      .sort((a, b) => b.score - a.score)
      .slice(0, 4);

    return {
      predictions,
      usedGPU: false
    };
  }
}
