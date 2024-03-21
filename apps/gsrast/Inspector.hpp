#pragma once

#include "AuxBuffer.cuh"
#include "DrawBase.hpp"
#include "FirstPersonCamera.hpp"
#include "GSRastWindow.hpp"
#include "PLYExplorer.hpp"
#include <map>

#define TNR ImGui::TableNextRow();
#define TNC ImGui::TableNextColumn();


/**
 * I am an SoA recording timestamp - frame rate
 */
struct FrameData
{
    std::vector<float> timestamps;
    std::vector<float> framerates;
};

/**
 * I am one single GeometryState (representing one Gaussian.)
 */
struct DownloadedGeometryState
{
    uint32_t tilesTouched;
    float depth; // Projected depths
    bool clamped;
    int internalRadius;
    glm::vec2 means2D;
    float cov3D[6];
    glm::vec4 conicOpacity;
    gscuda::gs::MathematicalEllipsoid ellipsoid;
    gscuda::gs::MathematicalEllipse ellipse;
    glm::vec3 rgb;
    uint32_t pointOffset;

    /**
       Actually, I am now more than one GeometryState; I will inspect
       other Gaussian-related stuffs as well, for example, their
       rotation, scaling, quaternions, etc.
    */
    glm::vec4 position;
    glm::vec4 quaternion;
    glm::mat3 rotation;
    glm::vec4 scaling;
};

/**
 * I am a simple struct recording camera poses.
 * I won't record auto-update values, including:
 * front, right, view, and perspective.
 */
struct CamPose
{
    glm::vec3 position;
    glm::vec3 ypr; // yaw pitch roll
    float sensitivity, speed, near, far, fov;
    bool invertUp;
};

/**
 * I am responsible for:
 * 1. the UI overlay so that I can inspect and change up things.
 * 2. interfacing with the window and its contents.
 * 3. change up the visualization data, as needed.
 * 4. recording & restoring scene-specific camera poses.
 * I require a GSRastWindow and its respective data.
 */
class Inspector : public DrawBase
{
public:
    CLASS_PTRS(Inspector)

    Inspector(GSRastWindow *rastWindow);
    virtual ~Inspector();

    virtual void draw() override;

protected:
    /**
     * I can change the main visualization mode of the RastWindow.
     * Beware that this is a huge change and will have lots of side
     * effects, and in addition, will take up some time.
     */
    void changeVisMode(VisMode newVisMode);

    bool startTable();
    void endTable();

    void loadCameraPoses();
    void saveCameraPose(const CamPose &cp);
    void screenshot();

    /**
     * I provide various APIs to inspect various sorts of data.
     */
    void inspectFloat(const char *key, float v);
    void editableFloat(const char *key, float *v, float min, float max);
    void inspectFloat2(const char *key, const float *v);
    void inspectFloat3(const char *key, const float *v);
    void inspectFloat4(const char *key, const float *v);
    void inspectBoolean(const char *key, bool v);
    void editableBoolean(const char *key, bool *v);
    void inspectBBox(const char *key, const BBox &bbox);
    void inspectInt(const char *key, int v);
    void inspectMat(int rows, int cols, const char *key, const float *v);
    void inspectEllipsoid(const char *key,
			  const gscuda::gs::MathematicalEllipsoid &ellipsoid);
    void inspectEllipse(const char *key,
			const gscuda::gs::MathematicalEllipse &ellipse);

    /**
       drawOverlay is split into two parts: inspector and
       interfaces. Inspectors allows the inspection of various properties
       of the scene (under different visualization modes.) Interfaces
       allows the interfacing with the scene, for example, scene
       manipulation, etc.
    */
    void drawOverlay();
    void drawInspector();
    void drawInterfaces();

    GSRastWindow *_rastWindow;
    bool _demoWindowShown;

    int _tableCounter;
    char _sprinted[512];

    float _timeElapsed;
    FrameData _frameData;
    std::map<std::string, CamPose> _camPoses;

    char _screenshotPath[512];
    int _selectedGeom;
    DownloadedGeometryState _downloaded;
    gscuda::gs::MathematicalEllipsoid _testEllipsoid;
    gscuda::gs::MathematicalEllipse _testEllipse;

    PLYExplorer::Ptr _plyExplorer;
    bool _showDirs;
};
