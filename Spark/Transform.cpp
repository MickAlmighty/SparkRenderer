#include "Transform.hpp"

#include <rttr/registration>

namespace spark
{
Transform::Transform() {}
}  // namespace spark

RTTR_REGISTRATION
{
    rttr::registration::class_<spark::Transform>("Transform")
        .constructor()(rttr::policy::ctor::as_object)
        .property("local", &spark::Transform::local)
        .property("world", &spark::Transform::world);
}
