#include "GSEllipsoids.hpp"
#include "apps/gsrast/SplatData.hpp"
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

bool GSEllipsoids::configureFromSplatData(const SplatData::Ptr &splatData, const ShaderBase::Ptr &shader)
{
    if (!splatData->isValid())
    {
        return false;
    }
    _splatData = splatData;
    _numInstances = splatData->getNumGaussians();
    _bbox = splatData->getBBox();
    _center = glm::vec3(0.0f);

    _numVerts = 36;
    configure(cube, _numVerts, sizeof(cube), shader);

    // configure SSBO: position
    _positionSSBO = generatePointsSSBO(splatData->getPositions());

    // configure SSBO: scale
    _scaleSSBO = generatePointsSSBO(splatData->getScales());

    // Configure SSBO: color
    std::vector<glm::vec4> colors;
    for (int i = 0; i < _numInstances; i++)
    {
        colors.push_back(glm::vec4(splatData->getSHs()[i].shs[0],
                                   splatData->getSHs()[i].shs[1],
                                   splatData->getSHs()[i].shs[2],
                                   1.0f));
    }
    _colorSSBO = generatePointsSSBO(colors);

    // Configure SSBO: quaternion
    // https://automaticaddison.com/how-to-convert-a-quaternion-to-a-rotation-matrix/
    // NOTE to self: if it gets too laggy in the future, convert this to a rotation matrix
    _quatSSBO = generatePointsSSBO(splatData->getRotations());

    _alphaSSBO = generatePointsSSBO(splatData->getOpacities());

    // The loaded model is somehow upside down; refer to GSPC for more detail
    // _model = glm::scale(_model, glm::vec3(-1.0f, -1.0f, 1.0f));

    return true;
}

void GSEllipsoids::draw()
{
    // The difference is here; how it is drawn.
    if (_shader) _shader->use(*this);
    else
    {
        std::cout << "No shader ???" << std::endl;
    }

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
    _splatData = nullptr;
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

