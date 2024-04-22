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

  cuda/cuda_accel.h
  cuda/cuda_accel.cpp
  cuda/cuda_mesh.h
  cuda/cuda_mesh.cpp
  cuda/cuda_buffer.h
  cuda/cuda_buffer.cpp
  cuda/cuda_texture.h
  cuda/optix_context.h
  cuda/cuda_scene.h
  
  util/sampling.h
  util/math.h
  util/exception.h
  util/type.h

  gui/display.h
  gui/display.cpp

  shader/test.cu
  OPTIONS -rdc true
  )



target_link_libraries( ${target_name}
  ${CUDA_LIBRARIES}
  )
