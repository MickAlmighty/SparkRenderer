#include "Animation.hpp"

#include "Clock.h"
#include "Spark.h"

namespace spark
{
void Animation::update()
{
    animationPlayer.update(getGameObject(), Clock::getDeltaTime());
}

void Animation::play(bool isLooped)
{
    animationPlayer.play(isLooped);
}

void Animation::pause()
{
    animationPlayer.pause();
}

void Animation::stop()
{
    animationPlayer.stop(getGameObject());
}

bool Animation::isPlaying() const
{
    return animationPlayer.isPlayingAnimation();
}

bool Animation::hasData() const
{
    return animationData != nullptr;
}

std::string Animation::getAnimationDataRelativePath() const
{
    return animationData ? animationData->getPath().string() : "";
}

void Animation::setAnimationDataByRelativePath(std::string path)
{
    animationData = Spark::get().getResourceLibrary().getResourceByRelativePath<resources::AnimationData>(path);
    animationPlayer.setAnimationData(animationData);
}

void Animation::drawUIBody()
{
    if(animationData)
    {
        ImGui::Text("%s", animationData->getPath().string().c_str());

        float tm = animationPlayer.getTimeMarker();
        ImGui::DragFloat("Preview", &tm, 0.001f, 0.0f, 1.0f);
        if(tm != animationPlayer.getTimeMarker())
        {
            animationPlayer.setTimeMarker(tm);
        }

        if(ImGui::Button("Play"))
        {
            play(animationPlayer.isAnimationLooped());
        }
        ImGui::SameLine();
        {
            bool isAnimationLooped = animationPlayer.isAnimationLooped();
            ImGui::Checkbox("isLooped", &isAnimationLooped);
            if(isAnimationLooped != animationPlayer.isAnimationLooped())
            {
                animationPlayer.setLooped(isAnimationLooped);
            }
        }
        ImGui::SameLine();
        if(ImGui::Button("Pause"))
        {
            pause();
        }
        ImGui::SameLine();
        if(ImGui::Button("Stop"))
        {
            stop();
        }
    }

    if(const auto animationDataOpt = SparkGui::selectAnimationDataByFilePicker(); animationDataOpt)
    {
        animationData = animationDataOpt.value();
        animationPlayer.setAnimationData(animationData);
    }
}
}  // namespace spark

RTTR_REGISTRATION
{
    rttr::registration::class_<spark::Animation>("Animation")
        .constructor()(rttr::policy::ctor::as_std_shared_ptr)
        .property("animationPlayer", &spark::Animation::animationPlayer)
        .property("animationData", &spark::Animation::getAnimationDataRelativePath, &spark::Animation::setAnimationDataByRelativePath);
}