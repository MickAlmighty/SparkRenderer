#include "Factory.h"
#include "Scene.h"
#include "GameObject.h"

using namespace spark;

std::shared_ptr<Scene> Factory::createScene(std::string&& name) {
    std::shared_ptr<Scene> scene(new Scene(std::move(name)));
    scene->getRoot()->setScene(scene);
    return scene;
}
