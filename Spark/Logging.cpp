#include "Logging.h"
#include <spdlog/sinks/basic_file_sink.h>
#include "spdlog/sinks/stdout_color_sinks.h"

std::shared_ptr<spark::logging::logger> spark::logging::getSparkLogger()
{
    static bool configure = true;
    if(configure)
    {
        configure = false;
        std::vector<spdlog::sink_ptr> sinks;
        auto consoleSink = std::make_shared<spdlog::sinks::stdout_color_sink_mt>();
        auto fileSink = std::make_shared<spdlog::sinks::basic_file_sink_mt>("spark.log", true);
        const spdlog::level::level_enum level{static_cast<spdlog::level::level_enum>(SPARK_ACTIVE_LEVEL)};
        const std::string pattern{"[%d.%m.%Y %T.%f] [%s:%#] [%t] [%^%l%$] %v"};
        consoleSink->set_level(level);
        fileSink->set_level(level);
        consoleSink->set_pattern(pattern);
        fileSink->set_pattern(pattern);
        sinks.push_back(consoleSink);
        sinks.push_back(fileSink);
        auto log = std::make_shared<spdlog::logger>("spark", std::begin(sinks), std::end(sinks));
        log->set_level(level);
        log->set_pattern(pattern);
        set_default_logger(log);
    }
    static std::shared_ptr<spdlog::logger> logger = spdlog::get("spark");
    return logger;
}

std::shared_ptr<spark::logging::logger> spark::logging::getRendererLogger()
{
    auto logger = spdlog::get(RENDERER_LOGGER_NAME);
    if(!logger)
    {
        logger = spdlog::basic_logger_mt(RENDERER_LOGGER_NAME, RENDERER_LOGGER_FILENAME, true);
        logger->set_pattern("%v");
        logger->set_level(static_cast<spdlog::level::level_enum>(SPARK_ACTIVE_LEVEL));
    }

    return logger;
}
