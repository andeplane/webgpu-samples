import { makeSample, SampleInit } from '../../components/SampleLayout';
import { mat4, vec3 } from 'wgpu-matrix';
import spriteWGSL from './sprite.wgsl';
import updateSpritesWGSL from './updateSprites.wgsl';

const init: SampleInit = async ({ canvas, pageState, gui }) => {
  const adapter = await navigator.gpu.requestAdapter();
  const device = await adapter.requestDevice();

  if (!pageState.active) return;
  const context = canvas.getContext('webgpu') as GPUCanvasContext;
  const devicePixelRatio = window.devicePixelRatio || 1;
  canvas.width = canvas.clientWidth * devicePixelRatio;
  canvas.height = canvas.clientHeight * devicePixelRatio;
  const presentationFormat = navigator.gpu.getPreferredCanvasFormat();

  context.configure({
    device,
    format: presentationFormat,
    alphaMode: 'premultiplied',
  });

  const uniformBuffer = device.createBuffer({
    label: 'Common.uniformBuffer',
    size:
      0 + //
      4 * 16 + // mvp
      4 * 16 + // inv_mvp
      4 * 4, // seed
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });

  const cameraBindGroupLayout = device.createBindGroupLayout({
    label: 'Camera.bindGroupLayout',
    entries: [
      {
        // common_uniforms
        binding: 0,
        visibility: GPUShaderStage.VERTEX | GPUShaderStage.COMPUTE,
        buffer: { type: 'uniform' },
      },
    ],
  });

  const cameraBindGroup = device.createBindGroup({
    label: 'Camera.bindGroup',
    layout: cameraBindGroupLayout,
    entries: [
      {
        // common_uniforms
        binding: 0,
        resource: {
          buffer: uniformBuffer,
          offset: 0,
          size: uniformBuffer.size,
        },
      },
    ],
  });

  const particleBindGroupLayout: GPUBindGroupLayout =
    device.createBindGroupLayout({
      label: 'Compute.bindGroupLayout',
      entries: [
        {
          binding: 0,
          visibility: GPUShaderStage.COMPUTE,
          buffer: { type: 'uniform' },
        },
        {
          binding: 1,
          visibility: GPUShaderStage.COMPUTE,
          buffer: { type: 'read-only-storage' },
        },
        {
          binding: 2,
          visibility: GPUShaderStage.COMPUTE,
          buffer: { type: 'storage' },
        },
      ],
    });

  const spriteShaderModule = device.createShaderModule({ code: spriteWGSL });
  const renderPipeline = device.createRenderPipeline({
    layout: device.createPipelineLayout({
      bindGroupLayouts: [cameraBindGroupLayout],
    }),
    vertex: {
      module: spriteShaderModule,
      entryPoint: 'vert_main',
      buffers: [
        {
          // instanced particles buffer
          arrayStride: 9 * 4,
          stepMode: 'instance',
          attributes: [
            {
              // instance position
              shaderLocation: 0,
              offset: 0,
              format: 'float32x3',
            },
            {
              // instance velocity
              shaderLocation: 1,
              offset: 3 * 4,
              format: 'float32x3',
            },
          ],
        },
        {
          // vertex buffer
          arrayStride: 2 * 4,
          stepMode: 'vertex',
          attributes: [
            {
              // vertex positions
              shaderLocation: 2,
              offset: 0,
              format: 'float32x2',
            },
          ],
        },
      ],
    },
    fragment: {
      module: spriteShaderModule,
      entryPoint: 'frag_main',
      targets: [
        {
          format: presentationFormat,
        },
      ],
    },
    primitive: {
      topology: 'triangle-list',
    },
  });

  const computeModule = device.createShaderModule({
    code: updateSpritesWGSL,
  });

  const computePipelineLayout = device.createPipelineLayout({
    bindGroupLayouts: [particleBindGroupLayout],
  });

  const calculateForcesPipeline = device.createComputePipeline({
    layout: computePipelineLayout,
    compute: {
      module: computeModule,
      entryPoint: 'calculateForces',
    },
  });

  const updatePositionsPipeline = device.createComputePipeline({
    layout: computePipelineLayout,
    compute: {
      module: computeModule,
      entryPoint: 'updatePositions',
    },
  });

  const updateVelocitiesPipeline = device.createComputePipeline({
    layout: computePipelineLayout,
    compute: {
      module: computeModule,
      entryPoint: 'updateVelocities',
    },
  });

  const renderPassDescriptor: GPURenderPassDescriptor = {
    colorAttachments: [
      {
        view: undefined, // Assigned later
        clearValue: { r: 0.0, g: 0.0, b: 0.0, a: 1.0 },
        loadOp: 'clear',
        storeOp: 'store',
      },
    ],
  };

  // prettier-ignore
  const vertexBufferData = new Float32Array([
    -0.01, -0.02,
    0.01, -0.02,
    0.0, 0.02,
  ]);

  const spriteVertexBuffer = device.createBuffer({
    size: vertexBufferData.byteLength,
    usage: GPUBufferUsage.VERTEX,
    mappedAtCreation: true,
  });
  new Float32Array(spriteVertexBuffer.getMappedRange()).set(vertexBufferData);
  spriteVertexBuffer.unmap();

  const scaleFactor = 500;
  const simParams = {
    deltaT: 0.01,
    epsilon: 1.0 / (scaleFactor * scaleFactor),
    sigma: 3.405 / scaleFactor,
    cutoff: (2.5 * 3.405) / scaleFactor,
    rotateCamera: false,
  };

  const simParamBufferSize = 4 * Float32Array.BYTES_PER_ELEMENT;
  const simParamBuffer = device.createBuffer({
    size: simParamBufferSize,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });

  function updateSimParams() {
    device.queue.writeBuffer(
      simParamBuffer,
      0,

      new Float32Array([
        simParams.deltaT,
        simParams.epsilon,
        simParams.sigma,
        simParams.cutoff,
      ])
    );
  }

  updateSimParams();
  Object.keys(simParams).forEach((k) => {
    gui.add(simParams, k).onFinishChange(updateSimParams);
  });

  function createFCC(
    systemLength: number,
    latticeConstant: number,
    particleData: Float32Array
  ) {
    const basisVectors = [
      [0, 0, 0],
      [0.5, 0.5, 0],
      [0.5, 0, 0.5],
      [0, 0.5, 0.5],
    ];
    const n = systemLength / latticeConstant;

    let particleIndex = 0;
    for (let i = 0; i < n; i++) {
      for (let j = 0; j < n; j++) {
        for (let k = 0; k < n; k++) {
          for (let v = 0; v < 4; v++) {
            const x = i + basisVectors[v][0];
            const y = j + basisVectors[v][1];
            const z = k + basisVectors[v][2];

            particleData[9 * particleIndex + 0] = x * latticeConstant;
            particleData[9 * particleIndex + 1] = y * latticeConstant;
            particleData[9 * particleIndex + 2] = z * latticeConstant;
            initialParticleData[9 * particleIndex + 3] =
              2 * (Math.random() - 0.5) * 0.1;
            initialParticleData[9 * particleIndex + 4] =
              2 * (Math.random() - 0.5) * 0.1;
            initialParticleData[9 * particleIndex + 5] =
              2 * (Math.random() - 0.5) * 0.1;

            particleIndex++;
          }
        }
      }
    }
    return particleIndex;
  }

  function createFccLattice(
    systemLength: number,
    latticeConstant: number,
    particleData: Float32Array
  ) {
    let particleIndex = 0;
    const numCells = systemLength / latticeConstant;
    for (let i = 0; i < numCells; i++) {
      for (let j = 0; j < numCells; j++) {
        const x = i * latticeConstant;
        const y = j * latticeConstant;
        particleData[9 * particleIndex + 0] = x;
        particleData[9 * particleIndex + 1] = y;
        initialParticleData[9 * particleIndex + 2] =
          2 * (Math.random() - 0.5) * 0.1;
        initialParticleData[9 * particleIndex + 3] =
          2 * (Math.random() - 0.5) * 0.1;
        particleIndex++;

        particleData[9 * particleIndex + 0] = x + latticeConstant / 2;
        particleData[9 * particleIndex + 1] = y + latticeConstant / 2;
        initialParticleData[9 * particleIndex + 2] =
          2 * (Math.random() - 0.5) * 0.1;
        initialParticleData[9 * particleIndex + 3] =
          2 * (Math.random() - 0.5) * 0.1;
        particleIndex++;

        particleData[9 * particleIndex + 0] = x + latticeConstant / 2;
        particleData[9 * particleIndex + 1] = y;
        initialParticleData[9 * particleIndex + 2] =
          2 * (Math.random() - 0.5) * 0.1;
        initialParticleData[9 * particleIndex + 3] =
          2 * (Math.random() - 0.5) * 0.1;
        particleIndex++;

        particleData[9 * particleIndex + 0] = x;
        particleData[9 * particleIndex + 1] = y + latticeConstant / 2;
        initialParticleData[9 * particleIndex + 2] =
          2 * (Math.random() - 0.5) * 0.1;
        initialParticleData[9 * particleIndex + 3] =
          2 * (Math.random() - 0.5) * 0.1;
        particleIndex++;
      }
    }
    return particleIndex;
  }

  const maxNumParticles = 256;
  const latticeConstant = 1.5;
  let initialParticleData = new Float32Array(maxNumParticles * 9);
  const numParticles = createFCC(2, latticeConstant, initialParticleData);
  console.log('Created ', numParticles, ' particles');
  initialParticleData = initialParticleData.slice(0, numParticles * 9);
  console.log('initialParticleData.byteLength', initialParticleData.byteLength);
  // for (let i = 0; i < numParticles; ++i) {
  //   initialParticleData[6 * i + 0] = 2 * (Math.random() - 0.5);
  //   initialParticleData[6 * i + 1] = 2 * (Math.random() - 0.5);
  //   initialParticleData[6 * i + 2] = 2 * (Math.random() - 0.5) * 0.1;
  //   initialParticleData[6 * i + 3] = 2 * (Math.random() - 0.5) * 0.1;
  //   initialParticleData[6 * i + 4] = 0;
  //   initialParticleData[6 * i + 5] = 0;
  // }

  // initialParticleData[6 * 0 + 0] = 0.0;
  // initialParticleData[6 * 0 + 1] = 0.0;
  // initialParticleData[6 * 1 + 0] = 0.8;
  // initialParticleData[6 * 1 + 1] = 0.8;

  const particleBuffers: GPUBuffer[] = new Array(2);
  const particleBindGroups: GPUBindGroup[] = new Array(2);

  for (let i = 0; i < 2; ++i) {
    particleBuffers[i] = device.createBuffer({
      size: initialParticleData.byteLength,
      usage: GPUBufferUsage.VERTEX | GPUBufferUsage.STORAGE,
      mappedAtCreation: true,
    });
    new Float32Array(particleBuffers[i].getMappedRange()).set(
      initialParticleData
    );
    particleBuffers[i].unmap();
  }

  for (let i = 0; i < 2; ++i) {
    particleBindGroups[i] = device.createBindGroup({
      layout: particleBindGroupLayout,
      entries: [
        {
          binding: 0,
          resource: {
            buffer: simParamBuffer,
          },
        },
        {
          binding: 1,
          resource: {
            buffer: particleBuffers[i],
            offset: 0,
            size: initialParticleData.byteLength,
          },
        },
        {
          binding: 2,
          resource: {
            buffer: particleBuffers[(i + 1) % 2],
            offset: 0,
            size: initialParticleData.byteLength,
          },
        },
      ],
    });
  }

  let t = 0;

  /** Updates the uniform buffer data */
  const update = (params: { rotateCamera: boolean; aspect: number }) => {
    const projectionMatrix = mat4.perspective(
      (2 * Math.PI) / 8,
      params.aspect,
      0.5,
      100
    );

    const viewRotation = params.rotateCamera ? t / 1000 : 0;

    const viewMatrix = mat4.lookAt(
      vec3.fromValues(
        Math.sin(viewRotation) * 3,
        0,
        Math.cos(viewRotation) * 3
      ),
      vec3.fromValues(0, 0, 0),
      vec3.fromValues(0, 1, 0)
    );
    mat4.inverse(viewMatrix, viewMatrix);
    const mvp = mat4.multiply(projectionMatrix, viewMatrix);
    const invMVP = mat4.invert(mvp);

    const uniformDataF32 = new Float32Array(uniformBuffer.size / 4);
    const uniformDataU32 = new Uint32Array(uniformDataF32.buffer);
    for (let i = 0; i < 16; i++) {
      uniformDataF32[i] = mvp[i];
    }
    for (let i = 0; i < 16; i++) {
      uniformDataF32[i + 16] = invMVP[i];
    }
    uniformDataU32[32] = 0xffffffff * Math.random();
    uniformDataU32[33] = 0xffffffff * Math.random();
    uniformDataU32[34] = 0xffffffff * Math.random();

    device.queue.writeBuffer(
      uniformBuffer,
      0,
      uniformDataF32.buffer,
      uniformDataF32.byteOffset,
      uniformDataF32.byteLength
    );
  };

  function frame() {
    // Sample is no longer the active page.
    if (!pageState.active) return;

    renderPassDescriptor.colorAttachments[0].view = context
      .getCurrentTexture()
      .createView();

    update({
      rotateCamera: simParams.rotateCamera,
      aspect: canvas.width / canvas.height,
    });

    const commandEncoder = device.createCommandEncoder();
    {
      const passEncoder = commandEncoder.beginComputePass();
      passEncoder.setPipeline(updateVelocitiesPipeline);
      passEncoder.setBindGroup(0, particleBindGroups[t % 2]);
      passEncoder.dispatchWorkgroups(Math.ceil(numParticles / 64));
      passEncoder.end();
    }
    {
      const passEncoder = commandEncoder.beginComputePass();
      passEncoder.setPipeline(updatePositionsPipeline);
      passEncoder.setBindGroup(0, particleBindGroups[t % 2]);
      passEncoder.dispatchWorkgroups(Math.ceil(numParticles / 64));
      passEncoder.end();
    }
    {
      const passEncoder = commandEncoder.beginComputePass();
      passEncoder.setPipeline(calculateForcesPipeline);
      passEncoder.setBindGroup(0, particleBindGroups[t % 2]);
      passEncoder.dispatchWorkgroups(Math.ceil(numParticles / 64));
      passEncoder.end();
    }
    {
      const passEncoder = commandEncoder.beginRenderPass(renderPassDescriptor);
      passEncoder.setPipeline(renderPipeline);
      passEncoder.setBindGroup(0, cameraBindGroup);
      passEncoder.setVertexBuffer(0, particleBuffers[(t + 1) % 2]);
      passEncoder.setVertexBuffer(1, spriteVertexBuffer);
      passEncoder.draw(3, numParticles, 0, 0);
      passEncoder.end();
    }
    device.queue.submit([commandEncoder.finish()]);

    ++t;
    requestAnimationFrame(frame);
  }
  requestAnimationFrame(frame);
};

const ComputeBoids: () => JSX.Element = () =>
  makeSample({
    name: 'Molecular dynamics simulation',
    description:
      'A GPU powered molecular dynamics simulation \
mimicing argon atoms in a gas.',
    gui: true,
    init,
    sources: [
      {
        name: __filename.substring(__dirname.length + 1),
        contents: __SOURCE__,
      },
      {
        name: 'updateSprites.wgsl',
        contents: updateSpritesWGSL,
        editable: true,
      },
      {
        name: 'sprite.wgsl',
        contents: spriteWGSL,
        editable: true,
      },
    ],
    filename: __filename,
  });

export default ComputeBoids;
