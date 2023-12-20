#version 430

layout (location = 0) in vec3 aPos;

layout (std430, binding = 0) buffer splatPosition {
    vec4 positions[];
};
layout (std430, binding = 1) buffer splatScale {
    vec4 scales[];
};
layout (std430, binding = 2) buffer splatColor {
    vec4 colors[];
};
layout (std430, binding = 3) buffer splatQuat {
    vec4 quats[];
};
layout (std430, binding = 4) buffer splatAlpha {
    float alphas[];
};

uniform mat4 model;
uniform mat4 view;
uniform mat4 perspective;

out vec3 position;
out vec3 ellipsoidCenter;
out vec3 ellipsoidScale;
out mat3 ellipsoidRot;
out float ellipsoidAlpha;

out vec3 color;

mat3 quatToMat3(vec4 q) {
  float qx = q.y;
  float qy = q.z;
  float qz = q.w;
  float qw = q.x;

  float qxx = qx * qx;
  float qyy = qy * qy;
  float qzz = qz * qz;
  float qxz = qx * qz;
  float qxy = qx * qy;
  float qyw = qy * qw;
  float qzw = qz * qw;
  float qyz = qy * qz;
  float qxw = qx * qw;

  return mat3(
    vec3(1.0 - 2.0 * (qyy + qzz), 2.0 * (qxy - qzw), 2.0 * (qxz + qyw)),
    vec3(2.0 * (qxy + qzw), 1.0 - 2.0 * (qxx + qzz), 2.0 * (qyz - qxw)),
    vec3(2.0 * (qxz - qyw), 2.0 * (qyz + qxw), 1.0 - 2.0 * (qxx + qyy))
  );
}

mat3 quatToMat3(vec4 q) {
  float qx = q.y;
  float qy = q.z;
  float qz = q.w;
  float qw = q.x;

  float qxx = qx * qx;
  float qyy = qy * qy;
  float qzz = qz * qz;
  float qxz = qx * qz;
  float qxy = qx * qy;
  float qyw = qy * qw;
  float qzw = qz * qw;
  float qyz = qy * qz;
  float qxw = qx * qw;

  return mat3(
    vec3(1.0 - 2.0 * (qyy + qzz), 2.0 * (qxy - qzw), 2.0 * (qxz + qyw)),
    vec3(2.0 * (qxy + qzw), 1.0 - 2.0 * (qxx + qzz), 2.0 * (qyz - qxw)),
    vec3(2.0 * (qxz - qyw), 2.0 * (qyz + qxw), 1.0 - 2.0 * (qxx + qyy))
  );
}

void main() {
    vec3 scale = vec3(scales[gl_InstanceID]) * 2.0;
    vec3 scaled = scale * aPos;
    mat3 rot = quatToMat(quats[gl_InstanceID]);

    vec3 rotated = rot * scaled;
    vec3 posOffset = rotated + vec3(positions[gl_InstanceID]);
    vec4 mPos = vec4(posOffset, 1.0);

    position = vec3(mPos);
    ellipsoidCenter = vec3(positions[gl_InstanceID]);
    ellipsoidScale = scale;
    ellipsoidRot = rot;
    ellipsoidAlpha = alphas[gl_InstanceID];

    gl_Position = perspective * view * model * mPos;
    color = vec3(colors[gl_InstanceID]) * 0.2 + vec3(0.5, 0.5, 0.5);
}
