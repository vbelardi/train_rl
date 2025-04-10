cmake_minimum_required(VERSION 3.10)
project(voxelgrid)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -march=native -flto")

# Find pybind11 and Eigen
find_package(pybind11 REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(yaml-cpp REQUIRED)




# Include directories for voxel grid, jps planner, and raycasting
include_directories(${EIGEN3_INCLUDE_DIR})
include_directories(${CMAKE_SOURCE_DIR}/src/multi_agent_pkgs/voxel_grid_util/include)
include_directories(${CMAKE_SOURCE_DIR}/src/multi_agent_pkgs/jps3d/include/jps_planner)
include_directories(${CMAKE_SOURCE_DIR}/src/multi_agent_pkgs/jps3d/include/jps_basis)
include_directories(${CMAKE_SOURCE_DIR}/src/multi_agent_pkgs/jps3d/include/jps_collision)

# Add the pybind11 module
pybind11_add_module(voxelgrid 
    voxelgrid_bindings.cpp
    src/multi_agent_pkgs/voxel_grid_util/src/voxel_grid.cpp
    src/multi_agent_pkgs/jps3d/src/jps_planner/jps_planner.cpp
    src/multi_agent_pkgs/jps3d/src/jps_planner/graph_search.cpp
    src/multi_agent_pkgs/jps3d/src/distance_map_planner/graph_search.cpp
    src/multi_agent_pkgs/jps3d/src/distance_map_planner/distance_map_planner.cpp
    src/multi_agent_pkgs/voxel_grid_util/src/raycast.cpp
)

# Ensure linking is done properly
target_include_directories(voxelgrid PRIVATE 
    ${CMAKE_SOURCE_DIR}/src/multi_agent_pkgs/voxel_grid_util/include
    ${CMAKE_SOURCE_DIR}/src/multi_agent_pkgs/jps3d/include/jps_planner/
    ${CMAKE_SOURCE_DIR}/src/multi_agent_pkgs/jps3d/include/jps_basis
    ${CMAKE_SOURCE_DIR}/src/multi_agent_pkgs/jps3d/include/jps_collision
)

# Link pybind11
target_link_libraries(voxelgrid PRIVATE pybind11::module)
target_link_libraries(voxelgrid PRIVATE yaml-cpp)
