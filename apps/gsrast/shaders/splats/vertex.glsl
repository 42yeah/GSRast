#version 430

layout (location = 0) in vec3 aPos;

layout (std430, binding = 0) buffer splatPosition {
    vec4 positions[];
};
layout (std430, binding = 1) buffer splatScale {
    vec4 scales[];
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
