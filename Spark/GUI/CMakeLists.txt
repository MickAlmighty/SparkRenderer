list(APPEND CURRENT_DIR_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/ImGuiFileBrowser.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/ImGuiFileBrowser.h
    ${CMAKE_CURRENT_SOURCE_DIR}/ImGuizmo.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/ImGuizmo.h
    ${CMAKE_CURRENT_SOURCE_DIR}/SparkGui.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/SparkGui.h
    ${CMAKE_CURRENT_SOURCE_DIR}/imgui_custom_widgets.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/imgui_custom_widgets.h
)

target_sources(Spark PRIVATE ${CURRENT_DIR_SOURCES})
target_sources(Spark PRIVATE ${IMGUI_SOURCE_LIST})