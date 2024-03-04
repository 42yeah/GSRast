#version 430 core

layout (location = 0) in vec3 aPos;
layout (location = 1) in vec3 aNormal;

uniform mat4 model;
uniform mat4 view;
uniform mat4 perspective;

out vec3 boxNormal;
out vec3 worldPos;
out mat4 ellipsoidModel;

void main() {
    vec4 wPos = model * vec4(aPos, 1.0);
    worldPos = vec3(wPos);
    boxNormal = vec3(model * vec4(aNormal, 0.0));
    gl_Position = perspective * view * wPos;
    ellipsoidModel = model;
}
