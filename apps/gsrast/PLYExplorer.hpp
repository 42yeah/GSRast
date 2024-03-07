#pragma once

#include <filesystem>
#include "Config.hpp"


struct PLYData
{
    bool isFolder;
    std::filesystem::path path;
    size_t size;
};

/**
 * I am in servitude to the Inspector; given a root
 * directory, I go through specific depths to list all the
 * PLY files within. Obviously I use filesystem. Obviously
 * I will elevate the whole project to use C++17. Why the hell not.
 */
class PLYExplorer
{
public:
    CLASS_PTRS(PLYExplorer)

    PLYExplorer();
    ~PLYExplorer();

    void setMaxLevel(int maxLevel);
    bool setBasePath(const std::filesystem::path &newPath);
    void listDirRecursive();

    int getMaxLevel() const;
    const std::vector<PLYData> &getPLYs() const;
    const std::filesystem::path &getBasePath() const;

protected:
    void listDir(const std::filesystem::path &path, int level, int maxLevel);

private:
    std::filesystem::path _basePath;
    std::vector<PLYData> _plys;
    int _maxLevel;
};
