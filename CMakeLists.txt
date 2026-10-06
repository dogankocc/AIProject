cmake_minimum_required (VERSION 3.10)

if (POLICY CMP0141)
  cmake_policy(SET CMP0141 NEW)
  set(CMAKE_MSVC_DEBUG_INFORMATION_FORMAT "$<IF:$<AND:$<C_COMPILER_ID:MSVC>,$<CXX_COMPILER_ID:MSVC>>,$<$<CONFIG:Debug,RelWithDebInfo>:EditAndContinue>,$<$<CONFIG:Debug,RelWithDebInfo>:ProgramDatabase>>")
endif()

project ("AIProject")

# Kaynak dosyalarını açıkça listele
set(SOURCES
    "${CMAKE_CURRENT_SOURCE_DIR}/main.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/core/Tensor.cpp"
)

# Executable oluştur
add_executable (AIProject ${SOURCES})

# Include dizinini hedefe özel ekle
target_include_directories(AIProject PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/include"
)

if (CMAKE_VERSION VERSION_GREATER 3.12)
  set_property(TARGET AIProject PROPERTY CXX_STANDARD 20)
endif()
