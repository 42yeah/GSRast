#include "PLYExplorer.hpp"
#include <cstring>


PLYExplorer::PLYExplorer() : _basePath(std::filesystem::current_path()) , _maxLevel(3)
{

}

PLYExplorer::~PLYExplorer()
{

}

void PLYExplorer::setMaxLevel(int maxLevel)
{
    _maxLevel = maxLevel;
}

bool PLYExplorer::setBasePath(const std::filesystem::path &newPath)
{
    if (std::filesystem::exists(newPath) &&
        std::filesystem::is_directory(newPath))
    {
        _basePath = newPath;
        return true;
    }
    return false;
}

void PLYExplorer::listDirRecursive()
{
    _plys.clear();
    if (_basePath.has_parent_path())
    {
        PLYData prevDir;
        prevDir.isFolder = true;
        prevDir.path = _basePath.parent_path();
        prevDir.size = 0;
        _plys.push_back(prevDir);
    }
    listDir(_basePath, 0, _maxLevel);
}

void PLYExplorer::listDir(const std::filesystem::path &path, int level, int maxLevel)
{
    if (!std::filesystem::is_directory(path) || level > maxLevel)
    {
        return;
    }
    for (const auto &entry : std::filesystem::directory_iterator(path, std::filesystem::directory_options::skip_permission_denied))
    {
        const std::filesystem::path &path = entry.path();
        if (std::filesystem::is_directory(path))
        {
            if (level == 0)
            {
                PLYData pd;
                pd.path = path;
                pd.isFolder = true;
                pd.size = 0;
                _plys.push_back(pd);
            }
            listDir(path, level + 1, maxLevel);
        }
        else if (path.has_extension())
        {
            char lower[64] = { 0 };
            const std::string &str = path.extension().string();
            for (int i = 1; i < str.size(); i++)
            {
                lower[i - 1] = tolower(str[i]);
            }
            if (wcsncmp(path.filename().c_str(), L"input.ply", sizeof(L"input.ply")) == 0)
            {
                continue; // Skip input PLYs
            }
            if (strncmp(lower, "ply", sizeof(lower)) == 0)
            {
                PLYData pd;
                pd.path = path;
                pd.isFolder = false;
                pd.size = std::filesystem::file_size(path);
                _plys.push_back(pd);
            }
        }
    }
}

int PLYExplorer::getMaxLevel() const
{
    return _maxLevel;
}

const std::vector<PLYData> &PLYExplorer::getPLYs() const
{
    return _plys;
}

const std::filesystem::path &PLYExplorer::getBasePath() const
{
    return _basePath;
}
