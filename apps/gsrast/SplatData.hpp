﻿#pragma once

#include "BBox.hpp"
#include "Config.hpp"
#include <vector>
#include <glm/glm.hpp>

// Spherical harmonics struct taken from GS source code.
template<int D>
struct SHs
{
    float shs[(D + 1) * (D + 1) * 3];
};

// Total number of floats:
// 3 + 3 + 48 + 1 + 3 + 4 = 62
struct RichPoint
{
    glm::vec3 position;
    glm::vec3 normal;
    SHs<3> shs;
    float opacity;
    glm::vec3 scale;
    glm::vec4 rotation;
};

struct GSSplats
{
    bool valid;
    int numSplats;
    std::vector<RichPoint> splats;
};


/**
 * I am the centralized representation of the splat data.
 * I am responsible for making sure that I can be passed onto
 * Gaussian renderers, regardless of my correctness state.
 * They will invoke my various getters, and I must not fail.
 * Hopefully, in the future I will take care of the GPU data as
 * well; so all visualization methods need not re-initialization
 * (and taking up multiple copies).
 */
class SplatData
{
public:
    CLASS_PTRS(SplatData)

    SplatData();
    SplatData(const std::string &plyPath);
    ~SplatData();

    bool loadFromPly(const std::string &plyPath);

    const glm::vec3 &getCenter() const;
    int getNumGaussians() const;
    const BBox &getBBox() const;

    /**
     * The positions and scales are vec4 due to the fact that
     * SSBO will err because of some weird memory alignment issues
     * (with vec3). I don't see this documented on the wiki but it
     * is what it is.
     */
    const std::vector<glm::vec4> &getPositions() const;
    const std::vector<glm::vec4> &getScales() const;
    const std::vector<SHs<3> > &getSHs() const;
    const std::vector<glm::vec4> &getRotations() const;
    const std::vector<float> &getOpacities() const;

    bool isValid() const;

protected:
    static std::unique_ptr<GSSplats> loadFromSplatsPly(const std::string &path);

    glm::vec3 _center;
    int _numGaussians;

    BBox _bbox;
    std::vector<glm::vec4> _positions;
    std::vector<glm::vec4> _scales;
    std::vector<SHs<3> > _shs;
    std::vector<glm::vec4> _rotations;
    std::vector<float> _opacities;

    bool _valid;
};
