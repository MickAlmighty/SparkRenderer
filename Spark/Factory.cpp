#include "Factory.h"
#include "Scene.h"
#include "GameObject.h"

using namespace spark;

std::shared_ptr<Scene> Factory::createScene(const std::string& name) {
    std::shared_ptr<Scene> scene(new Scene(name));
    scene->getRoot()->setScene(scene);
    return scene;
}
