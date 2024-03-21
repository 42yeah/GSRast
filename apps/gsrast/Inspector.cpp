#include "Inspector.hpp"
#include "AuxBuffer.cuh"
#include "Config.hpp"
#include "Database.hpp"
#include "DrawBase.hpp"
#include "FirstPersonCamera.hpp"
#include "Framebuffer.hpp"
#include "GSCuda.cuh"
#include "SplatData.hpp"
#include "apps/gsrast/GSGaussians.hpp"
#include "apps/gsrast/PLYExplorer.hpp"
#include <driver_types.h>
#include <imgui.h>
#include <lmdb.h>
#include <backends/imgui_impl_glfw.h>
#include <backends/imgui_impl_opengl3.h>
#include <cstdlib>
#include <cstring>
#include <glm/common.hpp>
#include <implot.h>
#include <memory>
#include <ctime>
#include <ostream>
#include <random>
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

    _plyExplorer = std::make_shared<PLYExplorer>();
    _showDirs = true;
    char basePath[512] = { 0 };
    if (Database::get()->get("plyexplorer", "__basepath", basePath, sizeof(basePath)) == MDB_SUCCESS)
    {
        _plyExplorer->setBasePath(basePath);
    }
    Database::get()->get("plyexplorer", "__showdirs", (char *) &_showDirs, sizeof(_showDirs));
    _plyExplorer->listDirRecursive();
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

    drawInspector();
    drawInterfaces();

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

void Inspector::drawInspector()
{
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
            float avg = 0.0f;
            int nContrib = 0;
            for (int i = (int) _frameData.framerates.size() - 1; i >= 0; i--)
            {
                if (_frameData.timestamps[i] <= _timeElapsed - 10.0f)
                {
                    break;
                }
                avg += _frameData.framerates[i];
                nContrib++;
            }
            if (nContrib != 0)
            {
                avg = avg / nContrib;
            }
            if (startTable())
            {
                inspectFloat("Delta time", (float) dt);
                inspectInt("FPS", (int) (1.0f / dt));
                inspectFloat("Average FPS", avg);
                inspectInt("#Frame data", (int) (_frameData.framerates.size()));
                endTable();
            }
            if (ImGui::Button("Clear frame data"))
            {
                _frameData.framerates.clear();
                _frameData.timestamps.clear();
            }
        }

        if (_rastWindow->getVisMode() == VisMode::Gaussians &&
            ImGui::CollapsingHeader("CUDA"))
        {
            ImGui::Text("Geometry buffer viewer");
            int ng = _rastWindow->getSplatData()->getNumGaussians();
            if (ng > 0)
            {
                GSGaussians::Ptr gaussian = _rastWindow->getRenderSelector()->getPtr<GSGaussians>();
                gscuda::gs::GeometryState geomState = gaussian->mapGeometryState();
                bool edited = ImGui::SliderInt("Select Gaussian", &_selectedGeom, 0, ng - 1);
                edited |= ImGui::InputInt("Input Gaussian", &_selectedGeom);
                if (ImGui::Button("Find next viewable Gaussian"))
                {
                    std::unique_ptr<uint32_t[]> tilesTouched(new uint32_t[ng]);
                    cudaMemcpy(tilesTouched.get(), &geomState.tilesTouched[0],
                               sizeof(uint32_t) * ng, cudaMemcpyDeviceToHost);
                    bool oob = false;
                    int i = _selectedGeom + 1;
                    for (; i < ng; i++)
                    {
                        if (tilesTouched[i] != 0)
                        {
                            break;
                        }
                    }
 
                    if (i >= ng)
                    {
                        for (i = 0; i < _selectedGeom; i++)
                        {
                            if (tilesTouched[i] != 0)
                            {
                                break;
                            }
                        }
                    }
                    _selectedGeom = i;
                    edited = true;
                }
                if (edited)
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
                    cudaMemcpy(&_downloaded.ellipsoid, &geomState.ellipsoids[_selectedGeom], sizeof(gscuda::gs::MathematicalEllipsoid), cudaMemcpyDeviceToHost);
                    cudaMemcpy(&_downloaded.ellipse, &geomState.ellipses[_selectedGeom], sizeof(gscuda::gs::MathematicalEllipse), cudaMemcpyDeviceToHost);

                    const glm::vec4 &rot = _rastWindow->getSplatData()->
                        getRotations()[_selectedGeom];
                    const glm::vec4 &scl = _rastWindow->getSplatData()->
                        getScales()[_selectedGeom];
                    const glm::vec4 &center = _rastWindow->getSplatData()->
                        getPositions()[_selectedGeom];
                    glm::mat3 rotMat(0.0f);
                    gscuda::quatToMatHost(&rotMat[0][0], &rot[0]);

                    _downloaded.position = center;
                    _downloaded.quaternion = rot;
                    _downloaded.rotation = rotMat;
                    _downloaded.scaling = scl;

                    // Need to update the selected geom in the
                    // forward params as well
                    gscuda::ForwardParams params = gaussian->getForwardParams();
                    params.selected = _selectedGeom;
                    gaussian->setForwardParams(params);
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
                    inspectEllipsoid("ellipsoid", _downloaded.ellipsoid);
                    inspectEllipse("ellipse", _downloaded.ellipse);
                    inspectFloat3("RGB", &_downloaded.rgb.x);
                    inspectInt("point offset", (int) _downloaded.pointOffset);
                    inspectFloat4("position", &_downloaded.position[0]);
                    inspectFloat4("quaternion", &_downloaded.quaternion[0]);
                    inspectMat(3, 3, "rotation", &_downloaded.rotation[0][0]);
                    inspectMat(4, 1, "scaling", &_downloaded.scaling[0]);

                    if (ImGui::Button("Test projection"))
                    {
                        FirstPersonCamera::Ptr fpCam = std::dynamic_pointer_cast<FirstPersonCamera>(_rastWindow->getCamera());
                        const glm::mat4 view = glm::transpose(fpCam->getView());

                        glm::mat3 normAxes;
                        normAxes = glm::mat3(fpCam->getFront(),
                                             glm::vec3(view[0]),
                                             glm::vec3(view[1]));

                        glm::vec3 planeCenter = fpCam->getPosition() + normAxes[0];
                        const float planeConstant = glm::dot(planeCenter, normAxes[0]);
                        const float projectedDistance = planeConstant -
                            glm::dot(fpCam->getPosition(), normAxes[0]);

                        gscuda::ellipsoidFromGaussianHost(&_testEllipsoid,
                                                          &_downloaded.rotation[0][0],
                                                          &_downloaded.scaling[0],
                                                          &_downloaded.position[0]);
                        glm::vec3 camPos = fpCam->getPosition();
                        gscuda::projectEllipsoidHost(&_testEllipse,
                                                     &_testEllipsoid,
                                                     &camPos[0],
                                                     &normAxes[0][0],
                                                     projectedDistance);
                    }
                    endTable();
                }

                if (startTable())
                {
                    inspectEllipsoid("Test ellipsoid",
                                     _testEllipsoid);
                    inspectEllipse("Test ellipse",
                                   _testEllipse);
                    endTable();
                }
            }

            if (ImGui::CollapsingHeader("Debug"))
            {
		if (ImGui::Button("Test adaptive OIT construction"))
		{
		    gscuda::MiniNode nodes[5];
		    int head, tail;
		    gscuda::initAdaptiveFHost(nodes, 5, head, tail);
		    std::uniform_real_distribution<float> distrib;
		    std::random_device dev;
		    for (int i = 0; i < 1000; i++)
		    {
			int id = (int) (distrib(dev) * 10000000);
			float depth = distrib(dev);
			float alpha = powf(distrib(dev), 3.0f);
			glm::vec3 color = glm::vec3(distrib(dev), distrib(dev), distrib(dev));
			gscuda::insertAdaptiveFHost(nodes, 5, depth, id, alpha,
						    &color[0], head, tail);
		    }

		    std::cout << "The done deal:" << std::endl;
		    int it = head;
		    while (it != -1)
		    {
			std::cout << nodes[it].depth << " " << nodes[it].alpha << " "
				  << nodes[it].color.r << " " << nodes[it].color.g << " "
				  << nodes[it].color.b << std::endl;
			it = nodes[it].next;
		    }
		}
		if (startTable())
		{
		    ImGuiIO &io = ImGui::GetIO();
		    const ImVec2 &mousePos = io.MousePos;
		    inspectFloat2("Mouse pos", &mousePos.x);
		    ImVec2 blockPos = { mousePos.x / BLOCK_X, (_rastWindow->getHeight() - mousePos.y) / BLOCK_Y };
		    inspectFloat2("Block pos", &blockPos.x);

		    GSGaussians::Ptr gaussian = _rastWindow->getRenderSelector()->getPtr<GSGaussians>();
		    gscuda::ForwardParams params = gaussian->getForwardParams();
		    params.highlightBlockX = blockPos.x;
		    params.highlightBlockY = blockPos.y;
		    gaussian->setForwardParams(params);

		    endTable();
		}
            }
        }
    }
    ImGui::End();

}

void Inspector::drawInterfaces()
{
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

        if (_rastWindow->getVisMode() == VisMode::Gaussians)
        {
            GSGaussians::Ptr gaussian = _rastWindow->getRenderSelector()->getPtr<GSGaussians>();
            gscuda::ForwardParams params = gaussian->getForwardParams();
            bool edited = ImGui::Checkbox("Multivariate Gaussian approximation", &params.cosineApprox);

            if (params.cosineApprox)
            {
                ImGui::SeparatorText("Cosine approximation params");
                edited |= ImGui::Checkbox("Show debug data", &params.debugCosineApprox);

                ImGui::Separator();
            }

            edited |= ImGui::Checkbox("Ellipsoid projection approximation", &params.ellipseApprox);
            if (params.ellipseApprox)
            {
                ImGui::SeparatorText("Ellip proj approx params");
                edited |= ImGui::SliderFloat("Focal dist", &params.ellipseApproxFocalDist, 1.0f, 10.0f);

                ImGui::Separator();
            }

            edited |= ImGui::Checkbox("Adaptive OIT", &params.adaptiveOIT);

            if (edited)
            {
                gaussian->setForwardParams(params);
            }
        }

        if (ImGui::CollapsingHeader("Switch scene"))
        {
            if (ImGui::Checkbox("Show directories", &_showDirs))
            {
                Database::get()->put("plyexplorer", "__showdirs", (char *) &_showDirs, sizeof(_showDirs));
            }
            ImGui::Text("Current path: %ws", _plyExplorer->getBasePath().c_str());
            if (startTable())
            {
                const PLYData *chosen = nullptr;
                for (const auto &ply : _plyExplorer->getPLYs())
                {
                    if (!_showDirs && ply.isFolder)
                    {
                        continue;
                    }
                    TNR TNC
                    if (ply.path.has_parent_path())
                    {
                        const std::filesystem::path &ppath = ply.path.parent_path();
                        ImGui::TextColored(ImVec4(0.5f, 0.5f, 0.5f, 1.0f), "%ws/", ppath.filename().c_str());
                        ImGui::SameLine();
                    }
                    ImGui::Text("%ws%s", ply.path.filename().c_str(), ply.isFolder ? "/" : "");
                    TNC
                    if (ply.isFolder)
                    {
                        snprintf(_sprinted, sizeof(_sprinted), "Change directory##%ws", ply.path.c_str());
                        if (ImGui::Button(_sprinted))
                        {
                            chosen = &ply;
                        }
                    }
                    else if (!ply.isFolder)
                    {
                        snprintf(_sprinted, sizeof(_sprinted), "Load##%ws", ply.path.c_str());
                        if (ImGui::Button(_sprinted))
                        {
                            if (_rastWindow->getSplatData()->loadFromPly(ply.path.string()))
                            {
                                _rastWindow->revisualize();
                                snprintf(message, sizeof(message), "Loaded %ws.", ply.path.c_str());
                                timeToLive = 3.0f;
                            }
                        }
                        ImGui::SameLine();
                        float size = ply.size / 1024.0f / 1024.0f;
                        ImGui::TextColored(ImVec4(0.5f, 0.5f, 0.5f, 1.0f), "(%.2fM)", size);
                    }
                }
                if (chosen)
                {
                    if (_plyExplorer->setBasePath(chosen->path))
                    {
                        Database::get()->put("plyexplorer", "__basepath", chosen->path.string().c_str(), chosen->path.string().size());
                        _plyExplorer->listDirRecursive();
                    }
                }
                endTable();
            }
        }

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
    }
    ImGui::End();
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
                int index = j * rows + i;
                ImGui::Text("%.6f", v[index]);
            }
        }
        ImGui::EndTable();
        ImGui::TreePop();
    }
}

void Inspector::inspectEllipsoid(const char *key,
                                 const gscuda::gs::MathematicalEllipsoid &ellipsoid)
{
    TNR TNC ImGui::Text("%s", key);
    snprintf(_sprinted, sizeof(_sprinted), "Inspector %d", _tableCounter++);
    TNC if (ImGui::TreeNode(_sprinted))
    {
        if (startTable())
        {
            inspectMat(3, 3, "A", &ellipsoid.A[0][0]);
            inspectFloat3("b", &ellipsoid.b[0]);
            inspectFloat("c", ellipsoid.c);
            endTable();
        }
        ImGui::TreePop();
    }
}

void Inspector::inspectEllipse(const char *key,
                               const gscuda::gs::MathematicalEllipse &ellipse)
{
    TNR TNC ImGui::Text("%s", key);
    snprintf(_sprinted, sizeof(_sprinted), "Inspector %d", _tableCounter++);
    TNC if (ImGui::TreeNode(_sprinted))
    {
        if (startTable())
        {
            inspectMat(2, 2, "A", &ellipse.A[0][0]);
            inspectFloat2("b", &ellipse.b.x);
            inspectFloat("c", ellipse.c);
            inspectFloat2("eigenvalues", &ellipse.eigenvalues[0]);
            inspectBoolean("degenerate", ellipse.degenerate);
            endTable();
        }
        ImGui::TreePop();
    }
}

