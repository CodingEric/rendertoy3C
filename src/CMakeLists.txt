OPTIX_add_sample_executable( optixPathTracer target_name
  wavefront.cpp
  shader/shader_common.h
  shader/shader_data.h
  shader/raygen.cu
  shader/miss.cu
  shader/closehit_radiance.cu

  mesh.h
  mesh.cpp
  wavefront.h
  light.h
  material.h
  bxdf.h
  
  util/sampling.h
  util/math.h

  gui/display.h
  gui/display.cpp

  shader/test.cu
  OPTIONS -rdc true
  )



target_link_libraries( ${target_name}
  ${CUDA_LIBRARIES}
  )
