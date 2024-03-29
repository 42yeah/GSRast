#pragma once

#define DEFAULT_WINDOW_W 1024
#define DEFAULT_WINDOW_H 768
#define WINDOW_TITLE "GSRaster"

#include <memory>
#include <iostream>
#include <stack>
#include <glm/glm.hpp>

#define CLASS_PTRS(CLASS_NAME) \
    using Ptr = std::shared_ptr<CLASS_NAME>;

// I recited them from my mind!
#define PI_F 3.1415f
#define PI 3.14159265
#define EPSILON 0.001f

// Some default camera params
#define DEFAULT_NEAR 0.001f
#define DEFAULT_FAR 100.0f
#define DEFAULT_FOV glm::radians(45.0f)

std::ostream &operator<<(std::ostream &os, const glm::vec2 &p);

std::ostream &operator<<(std::ostream &os, const glm::vec3 &p);

std::ostream &operator<<(std::ostream &os, const glm::vec4 &p);

std::ostream &operator<<(std::ostream &os, const glm::mat4 &m);

struct Viewport
{
    int x, y, w, h;
};

struct GlobalStacks
{
    std::stack<Viewport> viewportStack;
    std::stack<unsigned int> framebufferStack;
};

GlobalStacks &stacks();

#define NUM_CHANNELS 3 // Default 3, RGB
#define BLOCK_X 16
#define BLOCK_Y 16
