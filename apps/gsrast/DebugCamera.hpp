#pragma once

// Debug camera which copies SIBR_Viewers' camera configurations (by copying its data.)

#include "CameraBase.hpp"
#include "Config.hpp"

class DebugCamera : public CameraBase
{
public:
    CLASS_PTRS(DebugCamera)

    DebugCamera();

    virtual const glm::mat4 &getView() const override;
    virtual const glm::mat4 &getPerspective() const override;
    virtual const glm::vec3 &getPosition() const override;
    virtual const glm::vec3 &getFront() const override;
    virtual void update(const WindowBase &window) override;

protected:
    glm::mat4 _view;
    glm::mat4 _projection; // We ONLY have projection matrix
    glm::vec3 _eye;
};
