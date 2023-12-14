/*
 * Copyright (C) 2023, Inria
 * GRAPHDECO research group, https://team.inria.fr/graphdeco
 * All rights reserved.
 *
 * This software is free for non-commercial, research and evaluation use
 * under the terms of the LICENSE.md file.
 *
 * For inquiries contact sibr@inria.fr and/or George.Drettakis@inria.fr
 */


#version 430

uniform mat4 model;
uniform mat4 view;
uniform mat4 perspective;

layout (std430, binding = 0) buffer BoxCenters {
    vec4 positions[];
};
layout (std430, binding = 1) buffer Scales {
    vec4 scales[];
};
layout (std430, binding = 2) buffer Colors {
    vec4 colors[];
};
layout (std430, binding = 3) buffer Rotations {
    vec4 rotations[];
};
layout (std430, binding = 4) buffer Alphas {
    float alphas[];
};

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

const vec3 boxVertices[8] = vec3[8](
    vec3(-1, -1, -1),
    vec3(-1, -1,  1),
    vec3(-1,  1, -1),
    vec3(-1,  1,  1),
    vec3( 1, -1, -1),
    vec3( 1, -1,  1),
    vec3( 1,  1, -1),
    vec3( 1,  1,  1)
);

const int boxIndices[36] = int[36](
    0, 1, 2, 1, 3, 2,
    4, 6, 5, 5, 6, 7,
    0, 2, 4, 4, 2, 6,
    1, 5, 3, 5, 7, 3,
    0, 4, 1, 4, 5, 1,
    2, 3, 6, 3, 7, 6
);

out vec3 worldPos;
out vec3 ellipsoidCenter;
out vec3 ellipsoidScale;
out mat3 ellipsoidRotation;
out vec3 colorVert;
out mat4 MVP;
out flat int boxID;

void main() {
	boxID = gl_InstanceID;
    ellipsoidCenter = vec3(positions[boxID]);

	ellipsoidScale = vec3(scales[boxID]);
	ellipsoidScale = ellipsoidScale * 2.0;

	vec4 q = rotations[boxID];
	ellipsoidRotation = transpose(quatToMat3(q));

    int vertexIndex = boxIndices[gl_VertexID];
    worldPos = ellipsoidRotation * (ellipsoidScale * boxVertices[vertexIndex]);
    worldPos += ellipsoidCenter;

	colorVert = vec3(colors[boxID]) * 0.2 + 0.5;

    if (alphas[boxID] < 0.2) {
        worldPos = vec3(0.0);
        gl_Position = vec4(0.0);
    } else {
        MVP = perspective * view * model;
        gl_Position = MVP * vec4(worldPos, 1);
    }
}
