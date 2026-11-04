cmake_minimum_required (VERSION 3.10)

if (POLICY CMP0141)
  cmake_policy(SET CMP0141 NEW)
  set(CMAKE_MSVC_DEBUG_INFORMATION_FORMAT "$<$<CONFIG:Debug,RelWithDebInfo>:Embedded>")
endif()

project ("AIProject")

# Kaynak dosyalarını açıkça listele
set(SOURCES
    "${CMAKE_CURRENT_SOURCE_DIR}/main.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/core/Tensor.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/autograd/AutogradNode.cpp"
)

# Executable oluştur
add_executable (AIProject ${SOURCES})

# Include dizinini hedefe özel ekle
target_include_directories(AIProject PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/include"
)

add_executable(AIProjectTests
    "${CMAKE_CURRENT_SOURCE_DIR}/tests/TensorAutogradTests.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/core/Tensor.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/autograd/AutogradNode.cpp"
)

target_include_directories(AIProjectTests PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/include"
)

if (CMAKE_VERSION VERSION_GREATER 3.12)
  set_property(TARGET AIProjectTests PROPERTY CXX_STANDARD 20)
endif()

enable_testing()
add_test(NAME TensorAutogradTests COMMAND AIProjectTests)

if (CMAKE_VERSION VERSION_GREATER 3.12)
  set_property(TARGET AIProject PROPERTY CXX_STANDARD 20)
endif()
