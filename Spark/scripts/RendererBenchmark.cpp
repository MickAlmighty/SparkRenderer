#include "RendererBenchmark.hpp"

#include <ctime>
#include <sstream>

#include "Animation.hpp"
#include "Clock.h"
#include "GameObject.h"
#include "lights/PointLight.h"
#include "Logging.h"
#include "renderers/Renderer.hpp"
#include "Spark.h"

namespace spark::scripts
{
void RendererBenchmark::update()
{
    const auto animation = getGameObject()->getComponent<Animation>();

    if(animation && !animation->isPlaying() && benchmarkStarted)
    {
        stopBenchmark();
    }
}

void RendererBenchmark::drawUIBody()
{
    auto& spark = spark::Spark::get();
    const unsigned int width = spark.getRenderingContext().width, height = spark.getRenderingContext().height;

    for(const auto& [type, radioButtonName] : radioButtonsData)
    {
        if(ImGui::RadioButton(radioButtonName, spark.getRendererType() == type))
        {
            spark.selectRenderer(type, width, height);
        }
    }

    ImGui::DragInt("numberOfLights", &numberOfLights);
    if(numberOfLights < 0)
        numberOfLights = 0;

    if(ImGui::Button("Start Benchmark"))
    {
        startBenchmark();
    }
}

void RendererBenchmark::startBenchmark()
{
    const auto animation = getGameObject()->getComponent<Animation>();
    if(!animation || !animation->hasData())
    {
        return;
    }

    benchmarkStarted = true;

    auto& spark = spark::Spark::get();
    spark.isEditorEnabled = false;
    spark.getRenderer().isProfilingEnabled = true;
    selectedRenderer = spark.getRendererType();
    Clock::enableFixedDelta(1.0 / 60.0);

    generateGameObjectsWithLights();

    animation->play();
}

void RendererBenchmark::stopBenchmark()
{
    benchmarkStarted = false;

    getGameObject()->getComponent<Animation>()->stop();

    auto& spark = spark::Spark::get();
    spark.isEditorEnabled = true;
    spark.getRenderer().isProfilingEnabled = false;
    Clock::disableFixedDelta();

    releaseLights();
    spdlog::drop(spark::logging::RENDERER_LOGGER_NAME);

    try
    {
        const std::filesystem::path directory{"benchmarks"};
        if(!std::filesystem::exists(directory))
        {
            std::filesystem::create_directory(directory);
        }

        if(std::filesystem::exists(logging::RENDERER_LOGGER_FILENAME))
        {

            time_t now = time(0);
            tm* ltm = localtime(&now);

            std::stringstream time;
            time << "_" << ltm->tm_hour << "_" << ltm->tm_min << "_" << ltm->tm_sec;

            std::string rendererName = std::string(radioButtonsData.at(selectedRenderer));
            std::replace(std::begin(rendererName), std::end(rendererName), ' ', '_');

            const auto filename = rendererName + time.str() + ".csv";

            std::filesystem::rename(logging::RENDERER_LOGGER_FILENAME, directory / filename);
        }
    }
    catch(std::exception& e)
    {
        SPARK_ERROR(e.what());
    }
}

void RendererBenchmark::generateGameObjectsWithLights()
{
    if(numberOfLights != 0)
    {
        const auto scene = getGameObject()->getScene();
        lightContainer = scene->spawnGameObject("LightContainer");
        getGameObject()->getParent()->addChild(lightContainer.lock());

        for(int i = 0; i < numberOfLights; ++i)
        {
            const auto go = scene->spawnGameObject();
            go->setParent(lightContainer.lock());
            const auto light = go->addComponent<lights::PointLight>();

            light->setRadius(100);
            light->setColorStrength(200);
        }
    }
}

void RendererBenchmark::releaseLights() const
{
    if(numberOfLights != 0)
    {
        getGameObject()->getParent()->removeChild(lightContainer.lock());
    }
}
}  // namespace spark::scripts

RTTR_REGISTRATION
{
    rttr::registration::class_<spark::scripts::RendererBenchmark>("RendererBenchmark").constructor()(rttr::policy::ctor::as_std_shared_ptr);
}