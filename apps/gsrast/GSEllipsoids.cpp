#include "GSEllipsoids.hpp"
#include <gl/gl.h>

float cube[] = {
    // Back
    -0.5f, -0.5f, -0.5f,
    0.5f, -0.5f, -0.5f,
    0.5f, 0.5f, -0.5f,
    0.5f, 0.5f, -0.5f,
    -0.5f, 0.5f, -0.5f,
    -0.5f, -0.5f, -0.5f,

    // Front
    -0.5f, -0.5f, 0.5f,
    0.5f, -0.5f, 0.5f,
    0.5f, 0.5f, 0.5f,
    0.5f, 0.5f, 0.5f,
    -0.5f, 0.5f, 0.5f,
    -0.5f, -0.5f, 0.5f,

    // Bottom
    -0.5f, -0.5f, -0.5f,
    0.5f, -0.5f, -0.5f,
    0.5f, -0.5f, 0.5f,
    0.5f, -0.5f, 0.5f,
    -0.5f, -0.5f, 0.5f,
    -0.5f, -0.5f, -0.5f,

    // Top
    -0.5f, 0.5f, -0.5f,
    0.5f, 0.5f, -0.5f,
    0.5f, 0.5f, 0.5f,
    0.5f, 0.5f, 0.5f,
    -0.5f, 0.5f, 0.5f,
    -0.5f, 0.5f, -0.5f,

    // Left
    -0.5f, -0.5f, -0.5f,
    -0.5f, 0.5f, -0.5f,
    -0.5f, 0.5f, 0.5f,
    -0.5f, 0.5f, 0.5f,
    -0.5f, -0.5f, 0.5f,
    -0.5f, -0.5f, -0.5f,

    // Right
    0.5f, -0.5f, -0.5f,
    0.5f, 0.5f, -0.5f,
    0.5f, 0.5f, 0.5f,
    0.5f, 0.5f, 0.5f,
    0.5f, -0.5f, 0.5f,
    0.5f, -0.5f, -0.5f,
};

bool GSEllipsoids::configureFromPly(const std::string &path, ShaderBase::Ptr shader)
{
    const auto splatPtr = loadFromSplatsPly(path);
    if (!splatPtr)
    {
        return false;
    }
    _numInstances = splatPtr->numSplats;
    _bbox.reset();
    _center = glm::vec3(0.0f);

    _numVerts = 36;
    configure(cube, _numVerts, sizeof(cube), shader);

    // configure UBO: position
    {
        std::vector<glm::vec4> splatPosition;
        splatPosition.reserve(splatPtr->numSplats);
        for (int i = 0; i < splatPtr->splats.size(); i++)
        {
            const glm::vec3 &pos = splatPtr->splats[i].position;
            _bbox.enclose(pos);
            _center += pos;
            splatPosition.push_back(glm::vec4(pos.x, pos.y, pos.z, 1.0f));
        }
        if (splatPtr->splats.size() > 0)
        {
            _center /= splatPtr->splats.size();
        }
        _positionSSBO = generatePointsSSBO(splatPosition);
    }

    // configure UBO: scale
    {
        std::vector<glm::vec4> splatsScale;
        splatsScale.reserve(splatPtr->numSplats);
        for (int i = 0; i < splatPtr->splats.size(); i++)
        {
            const glm::vec3 &scale = glm::vec3(splatPtr->splats[i].scale);
            splatsScale.push_back(glm::vec4(exp(scale.x), exp(scale.y), exp(scale.z), 1.0f));
        }
        _scaleSSBO = generatePointsSSBO(splatsScale);
    }

    return true;
}

void GSEllipsoids::draw()
{
    // The difference is here; how it is drawn.
    if (_shader) _shader->use(*this);

    // Bind buffer bases.
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, _positionSSBO);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, _scaleSSBO);

    glBindVertexArray(_vao);
    glDrawArraysInstanced(GL_TRIANGLES, 0, _numVerts, _numInstances);
}

int GSEllipsoids::getNumInstances() const
{
    return _numInstances;
}

GSEllipsoids::GSEllipsoids() : GSPointCloud()
{
    _numInstances = 0;
    _positionSSBO = GL_NONE;
    _scaleSSBO = GL_NONE;
}

GSEllipsoids::~GSEllipsoids()
{
    if (_positionSSBO != GL_NONE)
    {
        glDeleteBuffers(1, &_positionSSBO);
    }
    if (_scaleSSBO != GL_NONE)
    {
        glDeleteBuffers(1, &_scaleSSBO);
    }
}

GLuint GSEllipsoids::generatePointsSSBO(const std::vector<glm::vec4> &points)
{
    GLuint ret = GL_NONE;
    glCreateBuffers(1, &ret);
    glNamedBufferStorage(ret, sizeof(glm::vec4) * points.size(), points.data(), GL_MAP_READ_BIT);

    return ret;
}
