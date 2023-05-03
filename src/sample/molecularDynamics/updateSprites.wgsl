struct Particle {
  pos : vec3<f32>,
  vel : vec3<f32>,
  force : vec3<f32>,
}
struct SimParams {
  deltaT : f32,
  epsilon : f32,
  sigma : f32,
  cutoff : f32,
}
struct Particles {
  particles : array<Particle>,
}
@binding(0) @group(0) var<uniform> params : SimParams;
@binding(1) @group(0) var<storage, read_write> particles : Particles;

@compute @workgroup_size(64)
fn calculateForces(@builtin(global_invocation_id) GlobalInvocationID : vec3<u32>) {
  var i = GlobalInvocationID.x;
  var posI = particles.particles[i].pos;
  var deltaPos : vec3<f32>;
  var force = vec3(0.0, 0.0, 0.0);
  var sigma6 = pow(params.sigma, 6.0);
  var epsilon24 = 24.0 * params.epsilon;
  
  for (var j = 0u; j < arrayLength(&particles.particles); j++) {
    if (j == i) {
      continue;
    }
    deltaPos = posI - particles.particles[j].pos;

    if (deltaPos.x > 1.0) {
      deltaPos.x -= 2.0;
    } else if (deltaPos.x < -1.0) {
      deltaPos.x += 2.0;
    }

    if (deltaPos.y > 1.0) {
      deltaPos.y -= 2.0;
    } else if (deltaPos.y < -1.0) {
      deltaPos.y += 2.0;
    }

    if (deltaPos.z > 1.0) {
      deltaPos.z -= 2.0;
    } else if (deltaPos.z < -1.0) {
      deltaPos.z += 2.0;
    }
    
    var r2 = dot(deltaPos, deltaPos);
    if (r2 > params.cutoff*params.cutoff || r2 < 0.0001) {
      continue;
    }
    var oneOverDr2 = 1.0/r2;
    var oneOverDr6 = oneOverDr2*oneOverDr2*oneOverDr2;
    force -= -epsilon24*sigma6*oneOverDr6*(2*sigma6*oneOverDr6 - 1)*oneOverDr2 * deltaPos;
  }
  // particles.particles[i].force = force;
}

@compute @workgroup_size(64)
fn updateVelocities(@builtin(global_invocation_id) GlobalInvocationID : vec3<u32>) { 
  var i = GlobalInvocationID.x;
  var vel = particles.particles[i].vel;
  var force = particles.particles[i].force;

  particles.particles[i].vel = vel + force * params.deltaT;
}

@compute @workgroup_size(64)
fn updatePositions(@builtin(global_invocation_id) GlobalInvocationID : vec3<u32>) { 
  var i = GlobalInvocationID.x;
  var pos = particles.particles[i].pos;
  var vel = particles.particles[i].vel;
  
  var newPos = pos + vel * params.deltaT;
  if (newPos.x > 1.0) {
    newPos.x -= 2.0;
  }
  if (newPos.x < -1.0) {
    newPos.x += 2.0;
  }
  if (newPos.y > 1.0) {
    newPos.y -= 2.0;
  }
  if (newPos.y < -1.0) {
    newPos.y += 2.0;
  }
  if (newPos.z > 1.0) {
    newPos.z -= 2.0;
  }
  if (newPos.z < -1.0) {
    newPos.z += 2.0;
  }

  particles.particles[i].pos = newPos;
}
