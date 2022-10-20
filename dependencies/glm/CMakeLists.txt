FetchContent_GetProperties(glm)
if(NOT glm_POPULATED)
  FetchContent_Populate(glm)
endif()

set(GLM_SOURCE_DIR ${glm_SOURCE_DIR} CACHE INTERNAL "")