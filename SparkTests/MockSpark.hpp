#pragma once
#include "pch.h"

#include "Spark.h"

namespace spark
{
class MockSpark : public Spark
{
    public:
    MOCK_METHOD(OpenGLContext&, getRenderingContext, (), (const, override));
    MOCK_METHOD(resourceManagement::ResourceLibrary&, getResourceLibrary, (), (const, override));
    MOCK_METHOD(SparkRenderer&, getRenderer, (), (const, override));
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