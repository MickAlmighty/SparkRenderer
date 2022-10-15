FetchContent_Declare(
    googletest
    GIT_REPOSITORY "https://github.com/google/googletest"
    GIT_TAG "release-1.12.1"
)

# For Windows: Prevent overriding the parent project's compiler/linker settings
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
set(INSTALL_GTEST OFF CACHE INTERNAL "" FORCE)
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
"MockSpark.hpp"
"GameObjectTest.cpp"
"GlHandle.cpp"
)

target_include_directories(SparkTests PUBLIC ${CMAKE_SOURCE_DIR}/Spark)
message(${googletest_SOURCE_DIR})
target_include_directories(SparkTests PUBLIC ${googletest_SOURCE_DIR}/googlemock/include)
target_include_directories(SparkTests PUBLIC ${googletest_SOURCE_DIR}/googletest/include)
#target_link_libraries(SparkTests googletest Spark)
target_link_libraries(SparkTests Spark)
#target_link_libraries(SparkTests googletest-distribution)
target_link_libraries(SparkTests debug ${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/libgtest_maind.a optimized gtest_main)
target_link_libraries(SparkTests debug ${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/libgmock_maind.a optimized gmock_main)

include(GoogleTest)
gtest_discover_tests(SparkTests)