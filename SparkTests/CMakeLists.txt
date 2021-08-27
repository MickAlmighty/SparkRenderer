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
"MockSpark.hpp" )

target_include_directories(SparkTests PUBLIC ${CMAKE_SOURCE_DIR}/Spark)
message(${googletest_SOURCE_DIR})
target_include_directories(SparkTests PUBLIC ${googletest_SOURCE_DIR}/googlemock/include)
target_include_directories(SparkTests PUBLIC ${googletest_SOURCE_DIR}/googletest/include)
#target_link_libraries(SparkTests googletest Spark)
target_link_libraries(SparkTests Spark)
target_link_libraries(SparkTests debug gtest_maind optimized gtest_main)
target_link_libraries(SparkTests debug gmock_maind optimized gmock_main)

include(GoogleTest)
gtest_discover_tests(SparkTests)