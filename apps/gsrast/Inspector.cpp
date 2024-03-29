﻿#include "Inspector.hpp"
#include "Database.hpp"
#include "DrawBase.hpp"
#include "FirstPersonCamera.hpp"
#include "Framebuffer.hpp"
#include "SplatData.hpp"
#include "apps/gsrast/GSGaussians.hpp"
#include "imgui.h"
#include <backends/imgui_impl_glfw.h>
#include <backends/imgui_impl_opengl3.h>
#include <cstring>
#include <glm/common.hpp>
#include <implot.h>
#include <memory>
#include <ctime>
#include <string.h>
#include <cuda_runtime.h>
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb_image_write.h>


Inspector::Inspector(GSRastWindow *rastWindow) : DrawBase("Inspector")
{
    _rastWindow = nullptr;
    if (rastWindow == nullptr || !rastWindow->valid() || !rastWindow->isImGuiInitialized())
    {
        return;
    }
    _rastWindow = rastWindow;
    _demoWindowShown = false;
    _tableCounter = 0;
    memset(_sprinted, 0, sizeof(_sprinted));
    _timeElapsed = 0.0f;

    // Create ImPlot context
    ImPlot::CreateContext();

    // Touch the database "cam_pose".
    Database::get()->put("cam_pose", "__gsrast", " ", 1);
    loadCameraPoses();

    memset(_screenshotPath, 0, sizeof(_screenshotPath));
    _screenshotPath[0] = '.';
    Database::get()->get("cam_pose", "__screenshotpath", _screenshotPath, sizeof(_screenshotPath));

    _selectedGeom = 0;
}

Inspector::~Inspector()
{
    ImPlot::DestroyContext();
}

void Inspector::draw()
{
    if (!_rastWindow || !_rastWindow->valid() || !_rastWindow->isImGuiInitialized())
    {
        return;
    }
    _timeElapsed += _rastWindow->deltaTime();
    _frameData.timestamps.push_back(_timeElapsed);
    _frameData.framerates.push_back(1.0f / _rastWindow->deltaTime());

    ImGui_ImplOpenGL3_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();

    drawOverlay();

    ImGui::Render();
    ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
}

void Inspector::drawOverlay()
{
    _tableCounter = 0;
    ImGuiIO &io = ImGui::GetIO();

    ImGui::SetNextWindowSize(ImVec2(490.0f, 252.0f), ImGuiCond_Appearing);
    ImGui::SetNextWindowPos(ImVec2(521.0f, 505.0f), ImGuiCond_Appearing);
    if (ImGui::Begin("Inspector"))
    {
        if (ImGui::CollapsingHeader("Meta"))
        {
            if (startTable())
            {
                ImVec2 winPos = ImGui::GetWindowPos();
                inspectFloat2("Window position", &winPos[0]);
                ImVec2 winSize = ImGui::GetWindowSize();
                inspectFloat2("Window size", &winSize[0]);
                endTable();
            }
        }

        if (ImGui::CollapsingHeader("Loaded data"))
        {
            const SplatData::Ptr &splatData = _rastWindow->getSplatData();
            if (startTable())
            {
                inspectInt("#Gaussians", splatData->getNumGaussians());
                inspectFloat3("Center", &splatData->getCenter().x);
                inspectBBox("BBox", splatData->getBBox());
                inspectBoolean("Valid", splatData->isValid());
                inspectInt("#Opacities", (int) splatData->getOpacities().size());
                inspectInt("#Positions", (int) splatData->getPositions().size());
                inspectInt("#Rotations", (int) splatData->getRotations().size());
                inspectInt("#SHs", (int) splatData->getSHs().size());
                inspectInt("#Scales", (int) splatData->getScales().size());
                inspectInt("Opacities size", (int) (sizeof(float) * splatData->getOpacities().size()));
                inspectInt("Positions size", (int) (sizeof(glm::vec4) * splatData->getPositions().size()));
                inspectInt("Rotations size", (int) (sizeof(glm::vec4) * splatData->getRotations().size()));
                inspectInt("SHs size", (int) (sizeof(SHs<3>) * splatData->getSHs().size()));
                inspectInt("Scales size", (int) (sizeof(glm::vec4) * splatData->getScales().size()));
                endTable();
            }
        }

        if (ImGui::CollapsingHeader("Camera"))
        {
            if (startTable())
            {
                FirstPersonCamera::Ptr fpCam = std::dynamic_pointer_cast<FirstPersonCamera>(_rastWindow->getCamera());
                glm::vec2 nearFar = fpCam->getNearFar();
                float spd = fpCam->getSpeed();
                inspectFloat3("Position", &fpCam->getPosition().x);
                inspectFloat3("YPR", &fpCam->getYPR().x);
                editableFloat("Speed", &spd, 0.01f, 100.0f);
                fpCam->setSpeed(spd);
                inspectFloat2("Clip", &nearFar.x);
                inspectFloat("FOV", fpCam->getFOV());
                bool invertUp = fpCam->getInvertUp();
                editableBoolean("Invert up", &invertUp);
                fpCam->setInvertUp(invertUp);
                inspectFloat3("Front", &fpCam->getFront().x);
                inspectFloat3("Right", &fpCam->getRight().x);
                inspectMat(4, 4, "View", (const float *) &fpCam->getView());
                inspectMat(4, 4, "Perspective", (const float *) &fpCam->getPerspective());
                endTable();
            }
        }

        if (ImGui::CollapsingHeader("Performances", ImGuiTreeNodeFlags_DefaultOpen))
        {
            double dt = _rastWindow->deltaTime();
            if (ImPlot::BeginPlot("##FPS hist", ImVec2(-1, 150)))
            {
                ImPlot::SetupAxes("time", "FPS");
                ImPlot::SetupAxisLimits(ImAxis_X1, _timeElapsed - 10.0f, _timeElapsed, ImGuiCond_Always);
                ImPlot::SetupAxisLimits(ImAxis_Y1, 0.0f, 200.0f);
                ImPlot::PlotLine("Frames per second", _frameData.timestamps.data(), _frameData.framerates.data(), (int) _frameData.framerates.size(), 0, 0, sizeof(float));
                ImPlot::EndPlot();
            }
            if (startTable())
            {
                inspectFloat("Delta time", (float) dt);
                inspectInt("FPS", (int) (1.0f / dt));
                inspectInt("#Frame data", (int) (_frameData.framerates.size()));
                endTable();
            }
            if (ImGui::Button("Clear frame data"))
            {
                _frameData.framerates.clear();
                _frameData.timestamps.clear();
            }
        }

        if (_rastWindow->getVisMode() == VisMode::Gaussians && ImGui::CollapsingHeader("CUDA"))
        {
            ImGui::Text("Geometry buffer viewer");
            int ng = _rastWindow->getSplatData()->getNumGaussians();
            if (ng > 0)
            {
                GSGaussians::Ptr gaussian = _rastWindow->getRenderSelector()->getPtr<GSGaussians>();
                gscuda::gs::GeometryState geomState = gaussian->mapGeometryState();
                bool edited = ImGui::SliderInt("Select Gaussian", &_selectedGeom, 0, ng - 1);
                edited |= ImGui::InputInt("Input Gaussian", &_selectedGeom);
                if (true || edited)
                {
                    _selectedGeom = glm::clamp(_selectedGeom, 0, ng - 1);
                    cudaMemcpy(&_downloaded.tilesTouched, &geomState.tilesTouched[_selectedGeom], sizeof(uint32_t), cudaMemcpyDeviceToHost);
                    cudaMemcpy(&_downloaded.depth, &geomState.depths[_selectedGeom], sizeof(float), cudaMemcpyDeviceToHost);
                    cudaMemcpy(&_downloaded.clamped, &geomState.clamped[_selectedGeom], sizeof(bool), cudaMemcpyDeviceToHost);
                    cudaMemcpy(&_downloaded.internalRadius, &geomState.internalRadii[_selectedGeom], sizeof(int), cudaMemcpyDeviceToHost);
                    cudaMemcpy(&_downloaded.means2D, &geomState.means2D[_selectedGeom], sizeof(glm::vec2), cudaMemcpyDeviceToHost);
                    cudaMemcpy(&_downloaded.cov3D, &geomState.cov3D[_selectedGeom * 6], sizeof(float) * 6, cudaMemcpyDeviceToHost);
                    cudaMemcpy(&_downloaded.conicOpacity, &geomState.conicOpacity[_selectedGeom], sizeof(glm::vec4), cudaMemcpyDeviceToHost);
                    cudaMemcpy(&_downloaded.rgb, &geomState.rgb[_selectedGeom], sizeof(glm::vec3), cudaMemcpyDeviceToHost);
                    cudaMemcpy(&_downloaded.pointOffset, &geomState.pointOffsets[_selectedGeom], sizeof(uint32_t), cudaMemcpyDeviceToHost);
                }
                glm::vec3 pos = _rastWindow->getSplatData()->getPositions()[_selectedGeom];
                if (ImGui::Button("Goto"))
                {
                    FirstPersonCamera::Ptr fpCam = std::dynamic_pointer_cast<FirstPersonCamera>(_rastWindow->getCamera());
                    fpCam->setPosition(pos - fpCam->getFront() * 1.0f);
                }
                if (startTable())
                {
                    inspectFloat3("position", &pos.x);
                    inspectInt("tiles touched", (int) _downloaded.tilesTouched);
                    inspectFloat("depth", _downloaded.depth);
                    inspectBoolean("clamped", _downloaded.clamped);
                    inspectInt("internal radius", (int) _downloaded.internalRadius);
                    inspectFloat2("means2D", &_downloaded.means2D[0]);
                    inspectMat(2, 3, "cov3D", _downloaded.cov3D);
                    inspectFloat4("conic opacity", &_downloaded.conicOpacity.x);
                    inspectFloat3("RGB", &_downloaded.rgb.x);
                    inspectInt("point offset", (int) _downloaded.pointOffset);
                    endTable();
                }
            }
        }

        ImGui::End();
    }

    ImGui::SetNextWindowSize(ImVec2(505.0f, 252.0f), ImGuiCond_Appearing);
    ImGui::SetNextWindowPos(ImVec2(13.0f, 505.0f), ImGuiCond_Appearing);
    if (ImGui::Begin("Interfaces"))
    {
        static char message[512] = { 0 };
        static float timeToLive = 0.0f;
        if (ImGui::Button("Take screen shot"))
        {
            // append a slash at the end, if it wasn't added already
            int pathLen = (int) strnlen(_screenshotPath, sizeof(_screenshotPath));
            if (pathLen == 0)
            {
                _screenshotPath[0] = '.';
                pathLen = 1;
            }
            if (_screenshotPath[pathLen - 1] != '/')
            {
                _screenshotPath[pathLen] = '/';
            }
            // dump the texture
            const Framebuffer::Ptr &fb = _rastWindow->getFramebuffer();
            std::unique_ptr<char[]> data(new char[fb->getWidth() * fb->getHeight() * 4]);
            glBindTexture(GL_TEXTURE_2D, fb->getTexture());
            glGetTexImage(GL_TEXTURE_2D, 0, GL_RGBA, GL_UNSIGNED_BYTE, data.get());
            // save to destination
            time_t now = time(0);
            tm lt;
            localtime_s(&lt, &now);
            char finalPath[512] = { 0 };
            strftime(_sprinted, sizeof(_sprinted), "%Y-%m-%d_%H-%M-%S.png", &lt);
            snprintf(finalPath, sizeof(finalPath), "%s%s", _screenshotPath, _sprinted);
            stbi_flip_vertically_on_write(true);
            if (!stbi_write_png(finalPath, fb->getWidth(), fb->getHeight(), 4, data.get(), 0))
            {
                snprintf(message, sizeof(message), "Cannot write to %s.", finalPath);
            }
            else
            {
                snprintf(message, sizeof(message), "Capture saved at %s.", finalPath);
            }
            timeToLive = 3.0f;
        }
        ImGui::SameLine();
        ImGui::SetNextItemWidth(128.0f);
        if (ImGui::InputText("Save path", _screenshotPath, sizeof(_screenshotPath)))
        {
            Database::get()->put("cam_pose", "__screenshotpath", _screenshotPath, sizeof(_screenshotPath));
        }
        if (timeToLive > 0.0f)
        {
            timeToLive -= _rastWindow->deltaTime();
            ImGui::TextColored(ImVec4(1.0f, 1.0f, 1.0f, timeToLive / 3.0f), "%s", message);
        }
        VisMode visMode = _rastWindow->getVisMode();
        switch (visMode)
        {
            case VisMode::PointCloud:
                snprintf(_sprinted, sizeof(_sprinted), "%s", "Point clouds");
                break;

            case VisMode::Ellipsoids:
                snprintf(_sprinted, sizeof(_sprinted), "%s", "Ellipsoids");
                break;

            case VisMode::Gaussians:
                snprintf(_sprinted, sizeof(_sprinted), "%s", "Gaussians");
                break;
        }
        if (ImGui::BeginCombo("Visualization", _sprinted))
        {
            if (ImGui::Selectable("Point clouds", visMode == VisMode::PointCloud))
            {
                changeVisMode(VisMode::PointCloud);
            }
            if (ImGui::Selectable("Ellipsoids", visMode == VisMode::Ellipsoids))
            {
                changeVisMode(VisMode::Ellipsoids);
            }
            if (ImGui::Selectable("Gaussians", visMode == VisMode::Gaussians))
            {
                changeVisMode(VisMode::Gaussians);
            }
            ImGui::EndCombo();
        }
        bool temp = false;
        ImGui::Checkbox("Multivariate Gaussian approximation", &temp);
        ImGui::Checkbox("SS-Ellipsoid projection approximation", &temp);
        ImGui::Checkbox("Sort caching", &temp);
        ImGui::SeparatorText("Saved poses");

        if (ImGui::Button("Save current pose", ImVec2(-1, 0)))
        {
            const FirstPersonCamera::Ptr &fpCam = std::dynamic_pointer_cast<FirstPersonCamera>(_rastWindow->getCamera());
            CamPose camPose;
            camPose.position = fpCam->getPosition();
            camPose.ypr = fpCam->getYPR();
            camPose.sensitivity = fpCam->getSensitivity();
            camPose.speed = fpCam->getSpeed();
            glm::vec2 nf = fpCam->getNearFar();
            camPose.near = nf.x;
            camPose.far = nf.y;
            camPose.fov = fpCam->getFOV();
            camPose.invertUp = fpCam->getInvertUp();
            saveCameraPose(camPose);
        }
        if (ImGui::BeginTable("camera_table", 2, ImGuiTableFlags_Resizable))
        {
            ImGui::TableSetupColumn("Camera", ImGuiTableColumnFlags_WidthFixed, 256);
            ImGui::TableSetupColumn("Operations", ImGuiTableColumnFlags_WidthStretch);
            ImGui::TableHeadersRow();
            for (auto cp = _camPoses.cbegin(); cp != _camPoses.cend();)
            {
                TNR TNC
                ImGui::Text("%s", cp->first.c_str());
                TNC
                snprintf(_sprinted, sizeof(_sprinted), "Select##%s", cp->first.c_str());
                if (ImGui::Button(_sprinted))
                {
                    const FirstPersonCamera::Ptr &fpCam = std::dynamic_pointer_cast<FirstPersonCamera>(_rastWindow->getCamera());
                    const CamPose &cpp = cp->second;
                    fpCam->setPosition(cpp.position);
                    fpCam->setYPR(cpp.ypr);
                    fpCam->setSensitivity(cpp.sensitivity);
                    fpCam->setSpeed(cpp.speed);
                    fpCam->setNearFar(cpp.near, cpp.far);
                    fpCam->setFOV(cpp.fov);
                    fpCam->setInvertUp(cpp.invertUp);
                }
                ImGui::SameLine();
                snprintf(_sprinted, sizeof(_sprinted), "Delete##%s", cp->first.c_str());
                if (ImGui::Button(_sprinted))
                {
                    auto old = cp++;
                    Database::get()->remove("cam_pose", old->first);
                    _camPoses.erase(old);
                }
                else
                {
                    cp++;
                }
            }
            endTable();
        }
        ImGui::End();
    }
    if (io.KeyCtrl && ImGui::IsKeyPressed(ImGuiKey_1))
    {
        _demoWindowShown = !_demoWindowShown;
    }
    if (_demoWindowShown)
    {
        ImGui::ShowDemoWindow(&_demoWindowShown);
        ImPlot::ShowDemoWindow();
    }
}

void Inspector::changeVisMode(VisMode newVisMode)
{
    switch (newVisMode)
    {
        case VisMode::PointCloud:
            _rastWindow->pointCloudMode();
            break;

        case VisMode::Ellipsoids:
            _rastWindow->ellipsoidsMode();
            break;

        case VisMode::Gaussians:
            _rastWindow->gaussianMode();
            break;
    }
}

bool Inspector::startTable()
{
    snprintf(_sprinted, sizeof(_sprinted), "Inspector#%d", _tableCounter++);
    bool shown = ImGui::BeginTable(_sprinted, 2, ImGuiTableFlags_Resizable);
    if (shown)
    {
        ImGui::TableSetupColumn("Key", ImGuiTableColumnFlags_WidthFixed);
        ImGui::TableSetupColumn("Value", ImGuiTableColumnFlags_WidthStretch);
        ImGui::TableHeadersRow();
    }
    return shown;
}

void Inspector::endTable()
{
    ImGui::EndTable();
}

void Inspector::loadCameraPoses()
{
    _camPoses.clear();
    CamPose camPose;
    Database::get()->iterate("cam_pose", [&](const std::string &key, MDB_val val) {
        assert(val.mv_size == sizeof(camPose));
        memcpy((char *) &camPose, val.mv_data, sizeof(camPose));
        _camPoses[key] = camPose;
    });
}

void Inspector::saveCameraPose(const CamPose &cp)
{
    // Generate an excerpt of the camera pose.
    snprintf(_sprinted, sizeof(_sprinted), "(%.2f, %.2f, %.2f) -> [%.2f, %.2f, %.2f]",
             cp.position.x, cp.position.y, cp.position.z,
             cp.ypr.x, cp.ypr.y, cp.ypr.z);
    _camPoses[_sprinted] = cp;
    Database::get()->put("cam_pose", _sprinted, (const char *) &cp, sizeof(cp));
}

void Inspector::inspectFloat(const char *key, float v)
{
    TNR TNC ImGui::Text("%s", key);
    TNC ImGui::Text("%f", v);
}

void Inspector::editableFloat(const char *key, float *v, float min, float max)
{
    TNR TNC ImGui::Text("%s", key);
    snprintf(_sprinted, sizeof(_sprinted), "slider##%s", key);
    TNC ImGui::SliderFloat(_sprinted, v, min, max);
    ImGui::InputFloat(_sprinted, v, 0.1f);
}

void Inspector::inspectFloat2(const char *key, const float *v)
{
    TNR TNC ImGui::Text("%s", key);
    TNC ImGui::Text("%.2f, %.2f", v[0], v[1]);
}

void Inspector::inspectFloat3(const char *key, const float *v)
{
    TNR TNC ImGui::Text("%s", key);
    TNC ImGui::Text("%.2f, %.2f, %.2f", v[0], v[1], v[2]);
}

void Inspector::inspectFloat4(const char *key, const float *v)
{
    TNR TNC ImGui::Text("%s", key);
    TNC ImGui::Text("%.2f, %.2f, %.2f, %.2f", v[0], v[1], v[2], v[3]);
}

void Inspector::inspectBoolean(const char *key, bool v)
{
    TNR TNC ImGui::Text("%s", key);
    TNC ImGui::Text("%s", v ? "true" : "false");
}

void Inspector::editableBoolean(const char *key, bool *v)
{
    TNR TNC ImGui::Text("%s", key);
    snprintf(_sprinted, sizeof(_sprinted), "checkbox##%s", key);
    TNC ImGui::Checkbox(_sprinted, v);
}

void Inspector::inspectInt(const char *key, int v)
{
    TNR TNC ImGui::Text("%s", key);
    TNC ImGui::Text("%d", v);
}

void Inspector::inspectBBox(const char *key, const BBox &bbox)
{
    const glm::vec3 &center = bbox.center;
    snprintf(_sprinted, sizeof(_sprinted), "BBox @ %.2f, %.2f, %.2f",
             center.x, center.y, center.z);
    TNR TNC ImGui::Text("%s", key);
    snprintf(_sprinted, sizeof(_sprinted), "Inspector#%d", _tableCounter++);
    TNC if (ImGui::TreeNode(_sprinted))
    {
        if (startTable())
        {
            inspectFloat3("BBox.min", &bbox.min.x);
            inspectFloat3("BBox.max", &bbox.max.x);
            inspectFloat3("BBox.center", &bbox.center.x);
            inspectBoolean("Bbox.infinite", bbox.infinite);
            endTable();
        }
        ImGui::TreePop();
    }
}

void Inspector::inspectMat(int rows, int cols, const char *key, const float *v)
{
    TNR TNC ImGui::Text("%s", key);
    snprintf(_sprinted, sizeof(_sprinted), "Inspector#%d", _tableCounter++);
    TNC if (ImGui::TreeNode(_sprinted))
    {
        snprintf(_sprinted, sizeof(_sprinted), "Inspector#%d", _tableCounter++);
        ImGui::BeginTable(_sprinted, cols + 1, ImGuiTableFlags_Resizable);
        for (int i = 0; i < cols + 1; i++)
        {
            if (i == 0)
            {
                snprintf(_sprinted, sizeof(_sprinted), "Rows");
            }
            else
            {
                snprintf(_sprinted, sizeof(_sprinted), "%d", i - 1);
            }
            ImGui::TableSetupColumn(_sprinted, ImGuiTableColumnFlags_WidthStretch);
        }
        ImGui::TableHeadersRow();
        for (int i = 0; i < rows; i++)
        {
            TNR TNC
            ImGui::Text("%d", i);
            for (int j = 0; j < cols; j++)
            {
                TNC
                int index = i * cols + j;
                ImGui::Text("%.6f", v[index]);
            }
        }
        ImGui::EndTable();
        ImGui::TreePop();
    }
}
