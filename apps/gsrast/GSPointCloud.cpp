#include "GSPointCloud.hpp"
#include "BufferGeo.hpp"
#include "apps/gsrast/SplatData.hpp"
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

using GSPC = GSPointCloud;

bool GSPointCloud::configureFromSplatData(const SplatData::Ptr &splatData, const ShaderBase::Ptr &shader)
{
    if (!splatData->isValid())
    {
        return false;
    }

    _splatData = splatData;
    _numVerts = splatData->getNumGaussians();
    _shader = shader;
    glBindVertexArray(_vao);
    glBindBuffer(GL_ARRAY_BUFFER, _vbo);
    const size_t posSize = sizeof(glm::vec4) * splatData->getNumGaussians();
    const size_t shSize = sizeof(SHs<3>) * splatData->getNumGaussians();
    glNamedBufferData(_vbo, (posSize + shSize), nullptr, GL_STATIC_DRAW);
    glNamedBufferSubData(_vbo, 0, posSize, _splatData->getPositions().data());
    glNamedBufferSubData(_vbo, posSize, shSize, _splatData->getSHs().data());

    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(glm::vec4), nullptr);

    // And now, spherical harmonics.
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(SHs<3>), (void *) (posSize));

    // Flip XY, for some reason
    // _model = glm::scale(_model, glm::vec3(-1.0f, -1.0f, 1.0f));

    return true;
}

void GSPointCloud::draw()
{
    if (_shader)
    {
        _shader->use(*this);
    }
    glBindVertexArray(_vao);
    glDrawArrays(GL_POINTS, 0, _numVerts);
}

const glm::vec3 &GSPointCloud::getCenter() const
{
    return _center;
}

GSPointCloud::GSPointCloud() : BufferGeo()
{
    _center = glm::vec3(0.0f);
    _splatData = nullptr;
}

GSPointCloud::~GSPointCloud()
{

}
