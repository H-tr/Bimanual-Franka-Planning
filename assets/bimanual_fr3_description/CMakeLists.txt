cmake_minimum_required(VERSION 3.10)
project(bimanual_fr3_description)

install(DIRECTORY meshes urdf rviz
  DESTINATION share/${PROJECT_NAME}
  OPTIONAL
)
