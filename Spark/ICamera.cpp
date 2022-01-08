#include "ICamera.hpp"

#include "CommonUtils.h"
#include "Spark.h"

namespace spark
{
glm::mat4 ICamera::getViewMatrix() const
{
    return glm::lookAt(position, cameraTarget, up);
}

glm::mat4 ICamera::getProjection() const
{
    return glm::perspectiveFov(glm::radians(fov), Spark::get().getRenderingContext().width * 1.0f, Spark::get().getRenderingContext().height * 1.0f,
                               zNear, zFar);
}

glm::mat4 ICamera::getProjectionReversedZInfiniteFarPlane() const
{
    return utils::getProjectionReversedZInfFar(Spark::get().getRenderingContext().width, Spark::get().getRenderingContext().height, fov, zNear);
}

glm::mat4 ICamera::getProjectionReversedZ() const
{
    return utils::getProjectionReversedZ(Spark::get().getRenderingContext().width, Spark::get().getRenderingContext().height, fov, zNear, zFar);
}
const UniformBuffer& ICamera::getUbo() const
{
    return cameraUbo;
}
}  // namespace spark

RTTR_REGISTRATION
{
    rttr::registration::class_<spark::ICamera>("ICamera")
        .constructor()(rttr::policy::ctor::as_std_shared_ptr)
        .property("cameraTarget", &spark::ICamera::cameraTarget)
        .property("position", &spark::ICamera::position)
        .property("front", &spark::ICamera::front)
        .property("up", &spark::ICamera::up)
        .property("right", &spark::ICamera::right)
        .property("fov", &spark::ICamera::fov)
        .property("zNear", &spark::ICamera::zNear)
        .property("zFar", &spark::ICamera::zFar);
}