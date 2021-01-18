cmake_minimum_required(VERSION 3.10)

set(PROJECT_NAME aff)
project(${PROJECT_NAME} LANGUAGES CXX VERSION 0.0.1 DESCRIPTION "Anime Face Filter")

set(CMAKE_VERBOSE_MAKEFILE TRUE)

# .lib
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/lib)

# .dll and .exe
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/bin) 

if (WIN32)
# option for Visual Studio
# -EHsc (enable proper Exxeption Handling) needs to avoid C4530
# -Wall is too noisy so that set -W4. 
# https://docs.microsoft.com/en-us/cpp/build/reference/compiler-option-warning-level?view=vs-2017
# "However, for a new project, it may be best to use /W4 in all compilations;
# this will ensure the fewest possible hard-to-find code defects."
set(CMAKE_CXX_FLAGS "/std:c++17 -W4 -EHsc")

else()
# g++ option for *nix
set(CMAKE_CXX_FLAGS  "-std=c++17 -Wall -O2")

endif()

get_directory_property(hasParent PARENT_DIRECTORY)
if(hasParent)
  message(STATUS "Has a parent scope.")
else()
  message(STATUS "Doesn't have a parent scope.")
endif()

function(set_with_parent ARG_NAME ARG_PATH ARG_TEXT)
    set(${ARG_NAME} ${ARG_PATH} CACHE PATH ${ARG_TEXT} FORCE)
    if (hasParent)
      set(${ARG_NAME} ${ARG_PATH} PARENT_SCOPE)
    endif()
endfunction(set_with_parent)

set(Aff_LIBS "aff")
set(Aff_INCLUDE_DIRS "${CMAKE_CURRENT_SOURCE_DIR};${CMAKE_CURRENT_SOURCE_DIR}/include;${CMAKE_CURRENT_SOURCE_DIR}/src")

find_package(OpenCV REQUIRED)
set(Aff_INCLUDE_DIRS ${Aff_INCLUDE_DIRS} ${OpenCV_INCLUDE_DIRS})
set(Aff_LIBS ${Aff_LIBS} ${OpenCV_LIBS})


if (NOT DEFINED DLIB_INSTALL_DIR)
  set(DLIB_INSTALL_DIR ${CMAKE_CURRENT_SOURCE_DIR}/third_party/dlib)
endif()
set(DLIB_ISO_CPP_ONLY ON)
add_subdirectory(${DLIB_INSTALL_DIR})
set(Dlib_INCLUDE_DIRS ${DLIB_INSTALL_DIR})
set(Aff_INCLUDE_DIRS ${Aff_INCLUDE_DIRS} ${Dlib_INCLUDE_DIRS})
set(Aff_LIBS ${Aff_LIBS} dlib)

# switch for optional libraries
option(AFF_USE_OPENMP "Use OpenMP to enable parallelization" ON)
message("AFF_USE_OPENMP: ${AFF_USE_OPENMP}")
if(AFF_USE_OPENMP)
  add_definitions(-DAFF_USE_OPENMP)
endif()

# For OpenMP
if(AFF_USE_OPENMP)
  find_package(OpenMP REQUIRED)
  if(OpenMP_FOUND)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
  endif()
endif()

set(Aff_LIB ${PROJECT_NAME})
add_library(${Aff_LIB}
  STATIC
  include/aff/core.h

  src/core.cpp
  src/dlib.cpp
  src/dlib.h
)

#set(Aff_LIBS ${Aff_LIBS} ${Aff_LIB)
#set(Aff_INCLUDE_DIRS ${Aff_INCLUDE_DIRS} ${Ugu_INCLUDE_DIRS})

set_with_parent(Aff_LIBS "${Aff_LIBS}" "Aff_LIBS")
set_with_parent(Aff_INCLUDE_DIRS "${Aff_INCLUDE_DIRS}" "Aff_INCLUDE_DIRS")

message("Aff_LIBS: ${Aff_LIBS}") 
message("Aff_INCLUDE_DIRS: ${Aff_INCLUDE_DIRS}") 

target_include_directories(${Aff_LIB} PUBLIC ${Aff_INCLUDE_DIRS})
set_target_properties(${Aff_LIB} PROPERTIES VERSION ${PROJECT_VERSION})

if (NOT hasParent)
  add_executable(demo app/demo.cpp)
  target_include_directories(demo PRIVATE ${Aff_INCLUDE_DIRS})
  target_link_libraries(demo
    ${Aff_LIBS}
    )
endif()

if (WIN32)
  set_property(DIRECTORY PROPERTY VS_STARTUP_PROJECT demo)
  # suppress C2338 for eigen
  # add_definitions(-D_ENABLE_EXTENDED_ALIGNED_STORAGE)
endif()
