#version 330 core

layout (location = 0) in vec3 aPos;
layout (location = 1) in vec3 aSH;

uniform mat4 model;
uniform mat4 view;
uniform mat4 perspective;

out vec3 color;

void main() {
    gl_Position = perspective * view * model * vec4(aPos, 1.0);
    color = aSH * 0.2 + vec3(0.5, 0.5, 0.5);
}
