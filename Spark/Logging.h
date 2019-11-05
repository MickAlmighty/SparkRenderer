#pragma once

#include <spdlog/spdlog.h>

#define SPARK_LEVEL_TRACE SPDLOG_LEVEL_TRACE
#define SPARK_LEVEL_DEBUG SPDLOG_LEVEL_DEBUG
#define SPARK_LEVEL_INFO SPDLOG_LEVEL_INFO
#define SPARK_LEVEL_WARN SPDLOG_LEVEL_WARN
#define SPARK_LEVEL_ERROR SPDLOG_LEVEL_ERROR
#define SPARK_LEVEL_CRITICAL SPDLOG_LEVEL_CRITICAL
#define SPARK_LEVEL_OFF SPDLOG_LEVEL_OFF

///
/// Defines active logging level of Spark
/// //todo: might want to make it changeable
///
#define SPARK_ACTIVE_LEVEL SPARK_LEVEL_DEBUG

#define SPARK_LOGGER_CALL(logger, level, ...) logger->log(spdlog::source_loc{__FILE__, __LINE__, SPDLOG_FUNCTION}, level, __VA_ARGS__)

#if SPARK_ACTIVE_LEVEL <= SPARK_LEVEL_TRACE
#define SPARK_LOGGER_TRACE(logger, ...) SPARK_LOGGER_CALL(logger, spdlog::level::trace, __VA_ARGS__)
#define SPARK_TRACE(...) SPARK_LOGGER_TRACE(spark::getSparkLogger(), __VA_ARGS__)
#else
#define SPARK_LOGGER_TRACE(logger, ...) (void)0
#define SPARK_TRACE(...) (void)0
#endif

#if SPARK_ACTIVE_LEVEL <= SPARK_LEVEL_DEBUG
#define SPARK_LOGGER_DEBUG(logger, ...) SPARK_LOGGER_CALL(logger, spdlog::level::debug, __VA_ARGS__)
#define SPARK_DEBUG(...) SPARK_LOGGER_DEBUG(spark::getSparkLogger(), __VA_ARGS__)
#else
#define SPARK_LOGGER_DEBUG(logger, ...) (void)0
#define SPARK_DEBUG(...) (void)0
#endif

#if SPARK_ACTIVE_LEVEL <= SPARK_LEVEL_INFO
#define SPARK_LOGGER_INFO(logger, ...) SPARK_LOGGER_CALL(logger, spdlog::level::info, __VA_ARGS__)
#define SPARK_INFO(...) SPARK_LOGGER_INFO(spark::getSparkLogger(), __VA_ARGS__)
#else
#define SPARK_LOGGER_INFO(logger, ...) (void)0
#define SPARK_INFO(...) (void)0
#endif

#if SPARK_ACTIVE_LEVEL <= SPARK_LEVEL_WARN
#define SPARK_LOGGER_WARN(logger, ...) SPARK_LOGGER_CALL(logger, spdlog::level::warn, __VA_ARGS__)
#define SPARK_WARN(...) SPARK_LOGGER_WARN(spark::getSparkLogger(), __VA_ARGS__)
#else
#define SPARK_LOGGER_WARN(logger, ...) (void)0
#define SPARK_WARN(...) (void)0
#endif

#if SPARK_ACTIVE_LEVEL <= SPARK_LEVEL_ERROR
#define SPARK_LOGGER_ERROR(logger, ...) SPARK_LOGGER_CALL(logger, spdlog::level::err, __VA_ARGS__)
#define SPARK_ERROR(...) SPARK_LOGGER_ERROR(spark::getSparkLogger(), __VA_ARGS__)
#else
#define SPARK_LOGGER_ERROR(logger, ...) (void)0
#define SPARK_ERROR(...) (void)0
#endif

#if SPARK_ACTIVE_LEVEL <= SPARK_LEVEL_CRITICAL
#define SPARK_LOGGER_CRITICAL(logger, ...) SPARK_LOGGER_CALL(logger, spdlog::level::critical, __VA_ARGS__)
#define SPARK_CRITICAL(...) SPARK_LOGGER_CRITICAL(spark::getSparkLogger(), __VA_ARGS__)
#else
#define SPARK_LOGGER_CRITICAL(logger, ...) (void)0
#define SPARK_CRITICAL(...) (void)0
#endif

namespace spark {
    using logger = spdlog::logger;
    static std::shared_ptr<logger> getSparkLogger();
}