#version 330 core

layout (location = 0) in vec3 aPos;

layout (std140) uniform splatPosition {
    vec4 positions[100000];
};
layout (std140) uniform splatScale {
    vec4 scales[100000];
};

uniform mat4 model;
uniform mat4 view;
uniform mat4 perspective;

out vec3 color;

void main() {
    vec3 scaled = vec3(scales[gl_InstanceID]) * aPos;
    vec3 posOffset = scaled + vec3(positions[gl_InstanceID]);
    gl_Position = perspective * view * model * vec4(posOffset, 1.0);
    color = vec3(1.0, 0.5, 0.0);
}
