FetchContent_GetProperties(dirent)
if(NOT dirent_POPULATED)
    FetchContent_Populate(dirent)
endif()

set(DIRENT_SOURCE_DIR ${dirent_SOURCE_DIR}/include CACHE INTERNAL "")