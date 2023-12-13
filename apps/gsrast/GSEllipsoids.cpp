#include "GSEllipsoids.hpp"
#include <glm/gtc/matrix_transform.hpp>

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

    std::vector<glm::vec4> points;
    points.resize(splatPtr->numSplats);

    // configure SSBO: position
    for (int i = 0; i < splatPtr->splats.size(); i++)
    {
        const glm::vec3 &pos = splatPtr->splats[i].position;
        _bbox.enclose(pos);
        _center += pos;
        points[i] = glm::vec4(pos.x, pos.y, pos.z, 1.0f);
    }
    if (splatPtr->splats.size() > 0)
    {
        _center /= splatPtr->splats.size();
    }
    _positionSSBO = generatePointsSSBO(points);

    // configure SSBO: scale
    for (int i = 0; i < splatPtr->splats.size(); i++)
    {
        const glm::vec3 &scale = splatPtr->splats[i].scale;
        points[i] = glm::vec4(exp(scale.x), exp(scale.y), exp(scale.z), 1.0f);
    }
    _scaleSSBO = generatePointsSSBO(points);

    // Configure SSBO: color
    for (int i = 0; i < splatPtr->splats.size(); i++)
    {
        const SHs<3> &shs = splatPtr->splats[i].shs;
        glm::vec4 color = glm::vec4(shs.shs[0], shs.shs[1], shs.shs[2], 1.0f);
        points[i] = color;
    }
    _colorSSBO = generatePointsSSBO(points);

    // Configure SSBO: quaternion
    // https://automaticaddison.com/how-to-convert-a-quaternion-to-a-rotation-matrix/
    // NOTE to self: if it gets too lag in the future, convert this to a rotation matrix
    for (int i = 0; i < splatPtr->splats.size(); i++)
    {
        points[i] = splatPtr->splats[i].rotation;
    }
    _quatSSBO = generatePointsSSBO(points);

    // The loaded model is somehow upside down; refer to GSPC for more detail
    _model = glm::scale(_model, glm::vec3(-1.0f, -1.0f, 1.0f));

    return true;
}

void GSEllipsoids::draw()
{
    // The difference is here; how it is drawn.
    if (_shader) _shader->use(*this);

    // Bind buffer bases.
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, _positionSSBO);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, _scaleSSBO);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, _colorSSBO);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, _quatSSBO);

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
    _colorSSBO = GL_NONE;
    _quatSSBO = GL_NONE;
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
    if (_colorSSBO != GL_NONE)
    {
        glDeleteBuffers(1, &_colorSSBO);
    }
    if (_quatSSBO != GL_NONE)
    {
        glDeleteBuffers(1, &_quatSSBO);
    }
}

GLuint GSEllipsoids::generatePointsSSBO(const std::vector<glm::vec4> &points)
{
    GLuint ret = GL_NONE;
    glCreateBuffers(1, &ret);
    glNamedBufferStorage(ret, sizeof(glm::vec4) * points.size(), points.data(), GL_MAP_READ_BIT);

    return ret;
}
