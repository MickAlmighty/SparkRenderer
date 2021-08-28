#pragma once
#include "pch.h"

#include "Spark.h"
#include "renderers/Renderer.hpp"

namespace spark
{
class MockSpark : public Spark
{
    public:
    MOCK_METHOD(OpenGLContext&, getRenderingContext, (), (const, override));
    MOCK_METHOD(resourceManagement::ResourceLibrary&, getResourceLibrary, (), (const, override));
    MOCK_METHOD(SceneManager&, getSceneManager, (), (const, override));
};

class SparkInstanceInjector
{
    public:
    static void injectInstance(Spark* instance)
    {
        Spark::ptr = instance;
    }
};
}  // namespace spark