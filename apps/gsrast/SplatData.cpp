#include "SplatData.hpp"
#include <fstream>
#include <glm/geometric.hpp>
#include <string>
#include <sstream>


float sigmoid(float x)
{
    return 1.0f / (1.0f + exp(-x));
}

SplatData::SplatData() : _center(0.0f), _numGaussians(0), _bbox(), _positions(), _scales(), _shs(), _rotations(), _opacities(), _valid(false)
{

}

SplatData::SplatData(const std::string &plyPath) : SplatData()
{
    loadFromPly(plyPath);
}

SplatData::~SplatData()
{

}

bool SplatData::loadFromPly(const std::string &plyPath)
{
    _center = glm::vec3(0.0f);
    _numGaussians = 0;
    _bbox.reset();
    _positions.clear();
    _scales.clear();
    _shs.clear();
    _rotations.clear();
    _opacities.clear();
    _valid = false;

    std::unique_ptr<GSSplats> splatPtr = loadFromSplatsPly(plyPath);
    if (!splatPtr->valid)
    {
        return false;
    }
    _bbox.reset();
    _center = glm::vec3(0.0f);
    _numGaussians = splatPtr->numSplats;
    for (const auto &sp : splatPtr->splats)
    {
        _positions.push_back(sp.position);
        _scales.push_back(glm::exp(sp.scale));
        _shs.push_back(sp.shs);
        _rotations.push_back(glm::normalize(sp.rotation));
        _opacities.push_back(sigmoid(sp.opacity));

        _bbox.enclose(sp.position);
        _center += sp.position;
    }
    if (splatPtr->splats.size() > 0)
    {
        _center /= splatPtr->splats.size();
    }

    _valid = true;
    return true;
}

const glm::vec3 &SplatData::getCenter() const
{
    return _center;
}

int SplatData::getNumGaussians() const
{
    return _numGaussians;
}

const BBox &SplatData::getBBox() const
{
    return _bbox;
}


const std::vector<glm::vec3> &SplatData::getPositions() const
{
    return _positions;
}

const std::vector<glm::vec3> &SplatData::getScales() const
{
    return _scales;
}

const std::vector<SHs<3> > &SplatData::getSHs() const
{
    return _shs;
}

const std::vector<glm::vec4> &SplatData::getRotations() const
{
    return _rotations;
}

const std::vector<float> &SplatData::getOpacities() const
{
    return _opacities;
}

bool SplatData::isValid() const
{
    return _valid;
}

std::unique_ptr<GSSplats> SplatData::loadFromSplatsPly(const std::string &path)
{
    std::unique_ptr<GSSplats> splats = std::make_unique<GSSplats>();
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
