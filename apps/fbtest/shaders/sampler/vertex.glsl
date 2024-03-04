#version 430 core

uniform mat4 view;
uniform mat4 perspective;

layout (location = 0) in vec3 aPos;

out vec2 uv;

void main() {
    gl_Position = perspective * view * vec4(aPos, 1.0);
    uv = aPos.xy * 0.5 + 0.5;
}
