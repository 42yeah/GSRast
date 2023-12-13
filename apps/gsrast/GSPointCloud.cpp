#include "GSPointCloud.hpp"
#include "BufferGeo.hpp"
#include <fstream>
#include <iostream>
#include <sstream>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

using GSPC = GSPointCloud;

bool GSPointCloud::configureFromPly(const std::string &path, ShaderBase::Ptr shader)
{
    const auto splatPtr = loadFromSplatsPly(path);
    if (!splatPtr->valid)
    {
        return false;
    }
    // initialize bounding box.
    _bbox.reset();
    _center = glm::vec3(0.0f);
    for (const auto &sp : splatPtr->splats)
    {
        _bbox.enclose(sp.position);
        _center += sp.position;
    }
    if (splatPtr->splats.size() > 0)
    {
        _center /= splatPtr->splats.size();
    }

    _numVerts = splatPtr->numSplats;
    _shader = shader;
    glBindVertexArray(_vao);
    glBindBuffer(GL_ARRAY_BUFFER, _vbo);
    glBufferData(GL_ARRAY_BUFFER, sizeof(GSPC::RichPoint) * splatPtr->numSplats, splatPtr->splats.data(), GL_STATIC_DRAW);

    constexpr int numFloats = 62; // Refer to GSPC.hpp for more details
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(float) * numFloats, nullptr);

    // And now, spherical harmonics.
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(float) * numFloats, (void *) (sizeof(float) * 6));

    // Flip XY, for some reason
    _model = glm::scale(_model, glm::vec3(-1.0f, -1.0f, 1.0f));

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
}

GSPointCloud::~GSPointCloud()
{

}

std::unique_ptr<GSPC::GSSplats> GSPointCloud::loadFromSplatsPly(const std::string &path)
{
    std::unique_ptr<GSPC::GSSplats> splats = std::make_unique<GSPC::GSSplats>();
    splats->numSplats = 0;
    splats->valid = false;

    std::ifstream reader(path, std::ios::binary);
    if (!reader.good())
    {
        std::cerr << "Bad PLY reader: " << path << "?" << std::endl;
        return std::move(splats);
    }

    // Parse PLY (from GaussianView:84 and so on)
    std::string buf;
    std::getline(reader, buf);
    std::getline(reader, buf);
    std::getline(reader, buf);
    std::stringstream ss(buf);
    std::string dummy;

    ss >> dummy >> dummy >> splats->numSplats;
    splats->splats.resize(splats->numSplats);
    std::cout << "Loading " << splats->numSplats << " splats.." << std::endl;

    while (std::getline(reader, dummy))
    {
        if (dummy.compare("end_header") == 0)
        {
            break;
        }
    }
    reader.read((char *) splats->splats.data(), splats->numSplats * sizeof(RichPoint));
    if (reader.eof())
    {
        std::cerr << "Reader is EOF?" << std::endl;
        splats->valid = false;
        return std::move(splats);
    }
    splats->valid = true;

    return std::move(splats);
}
