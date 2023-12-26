#version 430 core

in vec2 uv;
out vec4 color;

uniform int width;
uniform int height;

#define NUM_CHANNELS 3

// Output data SSBO
layout (std430, binding = 0) buffer image {
    float data[];
};

void main() {
    int x = int(uv.x * width);
    int y = int(uv.y * height);

    // Weird layout but OK
    float r = data[(y * width + x) + 0 * width * height];
    float g = data[(y * width + x) + 1 * width * height];
    float b = data[(y * width + x) + 2 * width * height];

    color = vec4(r, g, b, 1.0);
}
