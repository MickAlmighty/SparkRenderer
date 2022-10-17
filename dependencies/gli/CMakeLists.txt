FetchContent_GetProperties(gli)
if(NOT gli_POPULATED)
  FetchContent_Populate(gli)
endif()

set(GLI_SOURCE_DIR ${gli_SOURCE_DIR} CACHE INTERNAL "")