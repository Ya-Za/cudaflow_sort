cmake_minimum_required(VERSION 3.0.0)
project(test06 VERSION 0.1.0 LANGUAGES CXX CUDA)

add_executable(test06 main.cu)
target_include_directories(test06 PRIVATE "/home/yasin/codes/lib/taskflow")
target_link_libraries(test06 PRIVATE pthread)

target_compile_features(test06 PRIVATE cxx_std_17 cuda_std_17)
target_compile_options(test06 PRIVATE
  $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CXX_COMPILER_ID:GNU>>:-Wall -Wextra -Wfatal-errors>
  $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda -Xcompiler=-Wall,-Wextra,-Wfatal-errors>
)
