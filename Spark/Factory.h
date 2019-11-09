#pragma once
#include <memory>
#include <string>

namespace spark {
    class Scene;

    class Factory final {
    public:
        static std::shared_ptr<spark::Scene> createScene(std::string&& name);
    };

}
