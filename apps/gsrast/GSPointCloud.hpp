#pragma once

#include "BufferGeo.hpp"
#include "Config.hpp"
#include <string>
#include <vector>

class GSPointCloud : public BufferGeo
{
public:
    CLASS_PTRS(GSPointCloud)

    virtual bool configureFromPly(const std::string &path, ShaderBase::Ptr shader);

    virtual void draw() override;
    const glm::vec3 &getCenter() const;

    GSPointCloud();
    ~GSPointCloud();

protected:
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

    glm::vec3 _center;

    static std::unique_ptr<GSSplats> loadFromSplatsPly(const std::string &path);
};
