set(SOURCES
  edge_edge_2d.hpp
  edge_edge_2d.cpp
  edge_vert_2d.hpp
  edge_vert_2d.cpp
  vert_vert_2d.hpp
  vert_vert_2d.cpp
)

source_group(TREE "${CMAKE_CURRENT_SOURCE_DIR}" PREFIX "Source Files" FILES ${SOURCES})
target_sources(ipc_toolkit PRIVATE ${SOURCES})
