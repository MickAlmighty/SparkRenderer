FetchContent_GetProperties(stb)
if(NOT stb_POPULATED)
  FetchContent_Populate(stb)
endif()

set(STB_SOURCE_DIR ${stb_SOURCE_DIR} CACHE INTERNAL "")