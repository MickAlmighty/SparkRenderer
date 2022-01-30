list(APPEND CURRENT_DIR_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/ImGuiFileBrowser.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/ImGuiFileBrowser.h
    ${CMAKE_CURRENT_SOURCE_DIR}/ImGuizmo.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/ImGuizmo.h
    ${CMAKE_CURRENT_SOURCE_DIR}/SparkGui.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/SparkGui.h
)

target_sources(Spark PRIVATE ${CURRENT_DIR_SOURCES})

add_subdirectory(ImGui)