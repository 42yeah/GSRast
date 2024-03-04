#include "GSEllipsoids.hpp"
#include <glm/gtc/matrix_transform.hpp>

float cube[] = {
    // Back
    -1.0f, -1.0f, -1.0f,
    1.0f, -1.0f, -1.0f,
    1.0f, 1.0f, -1.0f,
    1.0f, 1.0f, -1.0f,
    -1.0f, 1.0f, -1.0f,
    -1.0f, -1.0f, -1.0f,

    // Front
    -1.0f, -1.0f, 1.0f,
    1.0f, -1.0f, 1.0f,
    1.0f, 1.0f, 1.0f,
    1.0f, 1.0f, 1.0f,
    -1.0f, 1.0f, 1.0f,
    -1.0f, -1.0f, 1.0f,

    // Bottom
    -1.0f, -1.0f, -1.0f,
    1.0f, -1.0f, -1.0f,
    1.0f, -1.0f, 1.0f,
    1.0f, -1.0f, 1.0f,
    -1.0f, -1.0f, 1.0f,
    -1.0f, -1.0f, -1.0f,

    // Top
    -1.0f, 1.0f, -1.0f,
    1.0f, 1.0f, -1.0f,
    1.0f, 1.0f, 1.0f,
    1.0f, 1.0f, 1.0f,
    -1.0f, 1.0f, 1.0f,
    -1.0f, 1.0f, -1.0f,

    // Left
    -1.0f, -1.0f, -1.0f,
    -1.0f, 1.0f, -1.0f,
    -1.0f, 1.0f, 1.0f,
    -1.0f, 1.0f, 1.0f,
    -1.0f, -1.0f, 1.0f,
    -1.0f, -1.0f, -1.0f,

    // Right
    1.0f, -1.0f, -1.0f,
    1.0f, 1.0f, -1.0f,
    1.0f, 1.0f, 1.0f,
    1.0f, 1.0f, 1.0f,
    1.0f, -1.0f, 1.0f,
    1.0f, -1.0f, -1.0f,
};

template<typename T>
GLuint generatePointsSSBO(const std::vector<T> &points)
{
    GLuint ret = GL_NONE;
    glCreateBuffers(1, &ret);
    glNamedBufferStorage(ret, sizeof(T) * points.size(), points.data(), GL_MAP_READ_BIT);

    return ret;
}

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
    std::vector<float> alphas;
    points.resize(splatPtr->numSplats);
    alphas.resize(splatPtr->numSplats);

    // configure SSBO: position
    for (int i = 0; i < splatPtr->splats.size(); i++)
    {
        glm::vec3 pos = splatPtr->splats[i].position;
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
    // NOTE to self: if it gets too laggy in the future, convert this to a rotation matrix
    for (int i = 0; i < splatPtr->splats.size(); i++)
    {
        points[i] = glm::normalize(splatPtr->splats[i].rotation);
    }
    _quatSSBO = generatePointsSSBO(points);

    for (int i = 0; i < splatPtr->splats.size(); i++)
    {
        alphas[i] = sigmoid(splatPtr->splats[i].opacity);
    }
    _alphaSSBO = generatePointsSSBO(alphas);

    // The loaded model is somehow upside down; refer to GSPC for more detail
    // _model = glm::scale(_model, glm::vec3(-1.0f, -1.0f, 1.0f));

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
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 4, _alphaSSBO);

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
    _alphaSSBO = GL_NONE;
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
    if (_alphaSSBO != GL_NONE)
    {
        glDeleteBuffers(1, &_alphaSSBO);
    }
}

