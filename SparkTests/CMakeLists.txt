FetchContent_Declare(
    googletest
    GIT_REPOSITORY "https://github.com/google/googletest"
    GIT_TAG "release-1.11.0"
)

# For Windows: Prevent overriding the parent project's compiler/linker settings
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

enable_testing()

add_executable(SparkTests "")
target_precompile_headers(SparkTests PRIVATE pch.h)
target_sources(SparkTests PRIVATE 
"pch.cpp"
"ObserverPatternTest.cpp"
"BufferTest.cpp"
"ReflectionTest.cpp"
"ResourceFactoryTest.cpp"
"ResourceIdentifierTest.cpp"
"SerializationTest.cpp"
)

add_custom_command(TARGET SparkTests POST_BUILD
                    COMMAND ${CMAKE_COMMAND} -E copy_directory
                        ${CMAKE_SOURCE_DIR}/res $<TARGET_FILE_DIR:SparkTests>/res)
                        
add_custom_command(TARGET SparkTests POST_BUILD
                    COMMAND ${CMAKE_COMMAND} -E copy_directory
                        ${CMAKE_SOURCE_DIR}/res ${CMAKE_BINARY_DIR}/res)

target_include_directories(SparkTests PUBLIC ${CMAKE_SOURCE_DIR}/Spark)
target_include_directories(SparkTests PUBLIC ${googletest_SOURCE_DIR}/googletest/include)
target_link_libraries(SparkTests gtest_main Spark)

include(GoogleTest)
gtest_discover_tests(SparkTests)