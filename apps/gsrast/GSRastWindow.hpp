#pragma once

#include "Config.hpp"
#include "Framebuffer.hpp"
#include "Window.hpp"
#include "PointCloudShader.hpp"
#include "SplatShader.hpp"
#include "CopyShader.hpp"
#include "SplatData.hpp"
#include "GSPointCloud.hpp"
#include "SamplerShader.hpp"
#include "apps/gsrast/RenderSelector.hpp"

class Inspector;

enum class VisMode
{
    PointCloud,
    Ellipsoids,
    Gaussians
};


/**
 * I am the main window. I manage the shaders, the rendering methods,
 * the inspectors, and so on. I can be interfaced with the inspector
 * (or something else.)
 */
class GSRastWindow : public Window
{
public:
    CLASS_PTRS(GSRastWindow)

    GSRastWindow();
    ~GSRastWindow();

    virtual void keyCallback(int key, int scancode, int action, int mods) override;

    const SplatData::Ptr &getSplatData() const;
    const PointCloudShader::Ptr &getPCShader() const;
    const SplatShader::Ptr &getSplatShader() const;
    const CopyShader::Ptr &getCopyShader() const;

    void pointCloudMode();
    void ellipsoidsMode();
    void gaussianMode();

    const Framebuffer::Ptr &getFramebuffer() const;
    const RenderSelector::Ptr &getRenderSelector() const;

    virtual void drawFrame() override;

    VisMode getVisMode() const;

protected:
    PointCloudShader::Ptr _pcShader;
    SplatShader::Ptr _splatShader;
    CopyShader::Ptr _copyShader;
    SplatData::Ptr _splatData;
    RenderSelector::Ptr _renderSelector;
    std::shared_ptr<Inspector> _inspector;
    VisMode _visMode;

    SamplerShader::Ptr _samplerShader;
    Framebuffer::Ptr _framebuffer;
};
