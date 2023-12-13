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

uniform mat4 model;
uniform mat4 view;
uniform mat4 perspective;

out vec3 color;

mat3 quatToMat(vec4 q) {
    return mat3(2.0 * (q.x * q.x + q.y * q.y) - 1.0, 2.0 * (q.y * q.z + q.x * q.w), 2.0 * (q.y * q.w - q.x * q.z), // 1st column
                2.0 * (q.y * q.z - q.x * q.w), 2.0 * (q.x * q.x + q.z * q.z) - 1.0, 2.0 * (q.z * q.w + q.x * q.y), // 2nd column
                2.0 * (q.y * q.w + q.x * q.z), 2.0 * (q.z * q.w - q.x * q.y), 2.0 * (q.x * q.x + q.w * q.w) - 1.0); // last column
}

void main() {
    vec3 scaled = vec3(scales[gl_InstanceID]) * aPos;
    mat3 rot = quatToMat(quats[gl_InstanceID]);
    vec3 rotated = rot * scaled;
    vec3 posOffset = rotated + vec3(positions[gl_InstanceID]);
    gl_Position = perspective * view * model * vec4(posOffset, 1.0);
    color = vec3(colors[gl_InstanceID]) * 0.2 + vec3(0.5, 0.5, 0.5);
}
