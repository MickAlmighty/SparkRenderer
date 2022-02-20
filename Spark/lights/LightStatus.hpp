#pragma once

namespace spark::lights
{
enum class LightCommand
{
    add,
    update,
    remove
};

template<typename T>
struct LightStatus
{
    LightCommand command{LightCommand::update};
    T* light{nullptr};
};
}  // namespace spark::lights