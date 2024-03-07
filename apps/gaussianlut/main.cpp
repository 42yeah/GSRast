#include <functional>
#include <vector>
#include <iostream>
#include <glm/glm.hpp>

float approxNorm2(float x)
{
    if (x >= 2.93f)
    {
        return glm::mix(0.2311f, 0.0f, glm::min(1.0f, x - 2.93f));
    }
    float sqrted = sqrtf(x);
    float a = (2.0f + cosf(sqrted)) * 0.33f;
    return a * a * a;
}

/**
 * A function LUT is needed to further accelerate.
 * Cosine LUTs are great because they are limited in range.
 */
std::function<float(float)> buildLUT(float a, float b, int n, std::function<float(float)> func)
{
    float step = (b - a) / n;
    std::vector<float> lut(n);
    for (int i = 0; i < n; i++)
    {
        lut[i] = func(a + step * i);
        std::cout << "Gen: " << (a + step * i) << " == " << lut[i] << std::endl;
    }
    return [&](float x) {
        if (x < a) return lut[0];
        if (x > b) return lut.back();
        int idx = (int) ((x - a) / step);
        int next_idx = idx + 1;
        std::cout << idx << std::endl;
        return lut[idx];
    };
}

int main()
{
    auto func = buildLUT(0.0f, 4.0f, 100, approxNorm2);
    return 0;
}
