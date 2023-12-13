#version 330 core

layout (location = 0) in vec3 aPos;

uniform mat4 model;
uniform mat4 view;
uniform mat4 perspective;

struct RichPoint {
    vec3 position;
    vec3 normal;
    float colors[48];
    float opacity;
    vec3 scale;
    vec4 rotation;
};

out vec3 color;

void main() {
    vec3 posOffset = aPos + vec3(float(gl_InstanceID) * 1.1, 0.0, 0.0);
    gl_Position = perspective * view * model * vec4(posOffset, 1.0);
    color = vec3(1.0, 0.5, 0.0);
}
