#pragma once

#include "Config.hpp"
#include "CameraBase.hpp"
#include <glm/glm.hpp>

class FirstPersonCamera : public CameraBase
{
public:
    CLASS_PTRS(FirstPersonCamera)

    /**
     * By default, when FirstPersonCamera was constructed,
     * it is situated at (0, 0, -1) looking at (0, 0, 1).
     */
    FirstPersonCamera();

    virtual const glm::mat4 &getView() const override;
    virtual const glm::mat4 &getPerspective() const override;
    virtual void update(const WindowBase &window) override;

    virtual void applyMotion(glm::vec3 dir);
    virtual void applyDelta(float dYaw, float dPitch);
    virtual void setSensitivity(float sensitivity);
    virtual const glm::vec3 &getYPR() const;
    virtual void setYPR(const glm::vec3 &ypr);
    virtual float getSpeed() const;
    virtual void setSpeed(float speed);
    virtual const glm::vec3 &getFront() const override;
    virtual const glm::vec3 &getRight() const;
    virtual const glm::vec3 &getPosition() const override;
    virtual void setPosition(const glm::vec3 &pos);
    virtual glm::vec2 getNearFar() const;
    void lookAt(const glm::vec3 &point);
    void setNearFar(float near, float far);
    float getFOV() const;
    float getSensitivity() const;
    void setFOV(float fov);
    void setInvertUp(bool invert);
    bool getInvertUp() const;

protected:
    glm::vec3 _position, _front, _right;
    glm::mat4 _view, _perspective;
    glm::vec3 _ypr; // yaw pitch roll
    float _sensitivity, _speed, _near, _far, _fov;
    bool _invertUp;
};
