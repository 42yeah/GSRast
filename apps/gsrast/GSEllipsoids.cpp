#include "GSEllipsoids.hpp"
#include <gl/gl.h>

bool GSEllipsoids::configureFromPly(const std::string &path, ShaderBase::Ptr shader)
{
    const auto splatPtr = loadFromSplatsPly(path);
    if (!splatPtr)
    {
        return false;
    }
    _numInstances = 10;

    // make a cube.
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

    _numVerts = 36;
    configure(cube, _numVerts, sizeof(cube), shader);

    // configure UBO: position
    {
        std::vector<glm::vec4> splatPosition;
        splatPosition.reserve(splatPtr->numSplats);
        for (int i = 0; i < splatPtr->splats.size(); i++)
        {
            const glm::vec3 &pos = splatPtr->splats[i].position;
            splatPosition.push_back(glm::vec4(pos.x, pos.y, pos.z, 1.0f));
        }
        _positionUBO = generatePointsUBO(splatPosition);
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
        _scaleUBO = generatePointsUBO(splatsScale);
    }

    return true;
}

void GSEllipsoids::draw()
{
    // The difference is here; how it is drawn.
    if (_shader) _shader->use(*this);

    // Bind buffer bases.
    glBindBufferBase(GL_UNIFORM_BUFFER, 0, _positionUBO);
    glBindBufferBase(GL_UNIFORM_BUFFER, 1, _scaleUBO);

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
    _positionUBO = GL_NONE;
    _scaleUBO = GL_NONE;
}

GSEllipsoids::~GSEllipsoids()
{
    if (_positionUBO != GL_NONE)
    {
        glDeleteBuffers(1, &_positionUBO);
    }
    if (_scaleUBO != GL_NONE)
    {
        glDeleteBuffers(1, &_scaleUBO);
    }
}

GLuint GSEllipsoids::generatePointsUBO(const std::vector<glm::vec4> &points)
{
    GLuint ret = GL_NONE;
    glGenBuffers(1, &ret);
    glBindBuffer(GL_UNIFORM_BUFFER, ret);
    glBufferData(GL_UNIFORM_BUFFER, sizeof(glm::vec4) * points.size(), points.data(), GL_STATIC_DRAW);

    return ret;
}
