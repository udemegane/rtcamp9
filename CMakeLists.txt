function(makeRelative FROM TO OUT)
    file(RELATIVE_PATH _TMP_STR "${FROM}" "${TO}")
    set(${OUT} "${_TMP_STR}" PARENT_SCOPE)
endfunction()

set(PROJECT_NAME rtcamp9_main)

set(USE_SLANG_RUNTIME 1)
set(USE_HLSL_RUNTIME 1)

# Executable
# get_filename_component(PROJECT_NAME ${CMAKE_CURRENT_SOURCE_DIR} NAME)
message(STATUS "-------------------------------")
message(STATUS "Processing Project ${PROJECT_NAME}")
add_executable(${PROJECT_NAME})

set_property(TARGET ${PROJECT_NAME} PROPERTY CXX_STANDARD 20)
set_property(TARGET ${PROJECT_NAME} PROPERTY CXX_STANDARD_REQUIRED ON)

SET(SAMPLE_FOLDER ${CMAKE_CURRENT_SOURCE_DIR}/${PROJECT_NAME})

# Sources for the project
file(GLOB SOURCE_FILES
    ${SAMPLE_FOLDER}/src/*.cpp
    ${SAMPLE_FOLDER}/src/*.hpp
    ${SAMPLE_FOLDER}/src/*.h
    ${SAMPLE_FOLDER}/shaders/*.hlsl
    ${SAMPLE_FOLDER}/shaders/*.hlsli
    ${SAMPLE_FOLDER}/shaders/*.h
)
target_sources(${PROJECT_NAME} PRIVATE ${SOURCE_FILES})
target_sources(${PROJECT_NAME} PRIVATE ${COMMON_SOURCE_FILES}) # Extra source from nvpro-core based on options
target_sources(${PROJECT_NAME} PRIVATE ${PACKAGE_SOURCE_FILES}) # Extra source from nvpro-core based on options
source_group(TREE ${SAMPLE_FOLDER} FILES ${SOURCE_FILES})
source_group("Other" FILES ${COMMON_SOURCE_FILES} ${PACKAGE_SOURCE_FILES})

# Readme
target_sources(${PROJECT_NAME} PRIVATE ${SAMPLE_FOLDER}/README.md)

# Include paths
target_include_directories(${PROJECT_NAME} PRIVATE ${SAMPLES_COMMON_DIR} ${SAMPLE_FOLDER})

# Linking with other libraries
target_link_libraries(${PROJECT_NAME}
    optimized ${LIBRARIES_OPTIMIZED}
    debug ${LIBRARIES_DEBUG}
    ${PLATFORM_LIBRARIES}
    nvpro_core
    ${UNIXLINKLIBS}
)

# Compile definitions
# the "config" directory doesn't really exist but serves as place holder
makeRelative("${OUTPUT_PATH}/config" "${SAMPLE_FOLDER}" TO_CURRENT_SOURCE_DIR)
makeRelative("${OUTPUT_PATH}/config" "${DOWNLOAD_TARGET_DIR}" TO_DOWNLOAD_TARGET_DIR)
target_compile_definitions(${PROJECT_NAME} PRIVATE PROJECT_NAME="${PROJECT_NAME}")
target_compile_definitions(${PROJECT_NAME} PRIVATE PROJECT_RELDIRECTORY="${TO_CURRENT_SOURCE_DIR}/")
target_compile_definitions(${PROJECT_NAME} PRIVATE PROJECT_DOWNLOAD_RELDIRECTORY="${TO_DOWNLOAD_TARGET_DIR}/")

# other properties
set_property(TARGET ${PROJECT_NAME} PROPERTY FOLDER "Samples")

# Using GLSL (by default)
if(USE_GLSL)
    # Vulkan shaders in project
    set(SHD_DIR ${SAMPLE_FOLDER}/shaders)
    file(GLOB SHD_HDR ${SHD_DIR}/*.glsl ${SHD_DIR}/*.h)
    file(GLOB SHD_SRC
        ${SHD_DIR}/*.vert
        ${SHD_DIR}/*.frag
        ${SHD_DIR}/*.tesc
        ${SHD_DIR}/*.tese
        ${SHD_DIR}/*.geom
        ${SHD_DIR}/*.comp
        ${SHD_DIR}/*.rgen
        ${SHD_DIR}/*.rchit
        ${SHD_DIR}/*.rmiss
        ${SHD_DIR}/*.rahit
        ${SHD_DIR}/*.rint
        ${SHD_DIR}/*.rcall
    )

    # Compiling shaders to Spir-V header
    compile_glsl(
        SOURCE_FILES ${SHD_SRC}
        HEADER_FILES ${SHD_HDR}
        DST "${SAMPLE_FOLDER}/_autogen"
        VULKAN_TARGET "vulkan1.3"
        HEADER ON
        DEPENDENCY ${VULKANSDK_BUILD_DEPENDENCIES}
        FLAGS -I${SHD_DIR} -I${NVPRO_CORE_DIR} -g -D__glsl
    )

    target_sources(${PROJECT_NAME} PRIVATE ${GLSL_SOURCES} ${GLSL_HEADERS})
    source_group(TREE ${SAMPLE_FOLDER} FILES ${GLSL_SOURCES} ${GLSL_HEADERS})
    target_compile_definitions(${PROJECT_NAME} PRIVATE USE_HLSL=0)
endif()

if(USE_HLSL)
    # Adding the HLSL header to the Visual Studio project
    file(GLOB HLSL ${SAMPLE_FOLDER}/shaders/*.hlsl ${SAMPLE_FOLDER}/shaders/*.hlsli ${SAMPLE_FOLDER}/shaders/*.h)
    target_sources(${PROJECT_NAME} PRIVATE ${HLSL})
    source_group("shaders" FILES ${HLSL})
    target_compile_definitions(${PROJECT_NAME} PRIVATE USE_HLSL=1)
endif()

if(USE_SLANG)
    # Adding the HLSL header to the Visual Studio project
    file(GLOB SLANG ${SAMPLE_FOLDER}/shaders/*.slang ${SAMPLE_FOLDER}/shaders/*.hlsli ${SAMPLE_FOLDER}/shaders/*.h)
    target_sources(${PROJECT_NAME} PRIVATE ${SLANG})
    source_group("shaders" FILES ${SLANG})
    target_compile_definitions(${PROJECT_NAME} PRIVATE USE_SLANG=1)
    target_compile_definitions(${PROJECT_NAME} PRIVATE USE_HLSL=0)
endif()

add_subdirectory(third_party)
set(THIRDPARTY_DIR ${CMAKE_CURRENT_SOURCE_DIR}/third_party)

if(USE_HLSL_RUNTIME)
    set(DXC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/third_party/dxc)
    add_library(dxc SHARED IMPORTED GLOBAL)
    set_target_properties(dxc PROPERTIES
        INTERFACE_INCLUDE_DIRECTORIES ${DXC_DIR}
        IMPORTED_IMPLIB ${DXC_DIR}/lib/x64/dxcompiler.lib
        IMPORTED_LOCATION ${DXC_DIR}/bin/x64/dxcompiler.dll
    )

    function(finalize_dxc_dependency thirdparty_root)
        # message("-- copy dxc dependencies")
        set(DXC_DIR ${thirdparty_root}/dxc)
        file(GLOB DXC_BIN LIST_DIRECTORIES false "${DXC_DIR}/bin/*" "${DXC_DIR}/lib/*")

        foreach(bin ${DXC_BIN})
            add_custom_command(
                TARGET ${PROJECT_NAME} POST_BUILD
                COMMAND ${CMAKE_COMMAND} ARGS -E copy ${bin} $<TARGET_FILE_DIR:${PROJECT_NAME}>
                COMMENT "Copy binary to $<TARGET_FILE_DIR:${PROJECT_NAME}>"
            )
        endforeach()
    endfunction()

    target_link_libraries(${PROJECT_NAME} dxc)

    set(SHADER_EXTENSION_REGEX "\.(hlsli|hlsl|h)$")

    function(copy_shaders shaders_dir)
        file(GLOB_RECURSE DXC_FILES "${shaders_dir}/*")

        add_custom_command(
            TARGET ${PROJECT_NAME} PRE_BUILD
            COMMAND ${CMAKE_COMMAND} -E make_directory $<TARGET_FILE_DIR:${PROJECT_NAME}>/shaders
            COMMENT "Make shader output directory..."
        )

        foreach(file ${DXC_FILES})
            if(${file} MATCHES ${SHADER_EXTENSION_REGEX})
                add_custom_command(
                    TARGET ${PROJECT_NAME} PRE_LINK
                    COMMAND ${CMAKE_COMMAND} ARGS -E copy_if_different ${file} $<TARGET_FILE_DIR:${PROJECT_NAME}>/shaders
                    COMMENT "Copying shader to ${file}"
                )
            endif()
        endforeach()
    endfunction()

    set(shader_source_dir ${CMAKE_CURRENT_SOURCE_DIR}/rtcamp9_main/shaders)
    copy_shaders(${shader_source_dir})

    finalize_dxc_dependency(${THIRDPARTY_DIR})
endif()

if(USE_SLANG_RUNTIME)
    set(SLANG_DIR ${CMAKE_CURRENT_SOURCE_DIR}/third_party/slang)
    add_library(slang SHARED IMPORTED GLOBAL)
    set_target_properties(slang PROPERTIES
        INTERFACE_INCLUDE_DIRECTORIES ${SLANG_DIR}
        IMPORTED_IMPLIB ${SLANG_DIR}/bin/windows-x64/release/slang.lib
        IMPORTED_LOCATION ${SLANG_DIR}/bin/windows-x64/release/slang.dll
    )
    file(GLOB SLANG_BIN LIST_DIRECTORIES false "${SLANG_DIR}/bin/windows-x64/release/*.dll" "${SLANG_DIR}/bin/windows-x64/release/*.lib")

    function(finalize_slang_dependency thirdparty_root)
        # message("-- Copy slang dependencies")
        set(SLANG_DIR ${thirdparty_root}/slang)
        file(GLOB SLANG_BIN LIST_DIRECTORIES false "${SLANG_DIR}/bin/windows-x64/release/*.dll" "${SLANG_DIR}/bin/windows-x64/release/*.lib")

        foreach(bin ${SLANG_BIN})
            add_custom_command(
                TARGET ${PROJECT_NAME} POST_BUILD
                COMMAND ${CMAKE_COMMAND} ARGS -E copy ${bin} $<TARGET_FILE_DIR:${PROJECT_NAME}>
                COMMENT "Copy binary to $<TARGET_FILE_DIR:${PROJECT_NAME}>"
            )
        endforeach()
    endfunction()

    target_link_libraries(${PROJECT_NAME} slang)

    set(SHADER_EXTENSION_REGEX "\.(slang|slangh|hlsli|hlsl|h)$")

    function(copy_shaders shaders_dir)
        file(GLOB_RECURSE SLANG_FILES "${shaders_dir}/*")

        add_custom_command(
            TARGET ${PROJECT_NAME} PRE_BUILD
            COMMAND ${CMAKE_COMMAND} -E make_directory $<TARGET_FILE_DIR:${PROJECT_NAME}>/shaders
            COMMENT "Make shader output directory..."
        )

        foreach(file ${SLANG_FILES})
            if(${file} MATCHES ${SHADER_EXTENSION_REGEX})
                add_custom_command(
                    TARGET ${PROJECT_NAME} POST_BUILD
                    COMMAND ${CMAKE_COMMAND} ARGS -E copy_if_different ${file} $<TARGET_FILE_DIR:${PROJECT_NAME}>/shaders
                    COMMENT "Copying shader to ${file}"
                )
            endif()
        endforeach()
    endfunction()

    set(shader_source_dir ${CMAKE_CURRENT_SOURCE_DIR}/rtcamp9_main/shaders)
    copy_shaders(${shader_source_dir})

    finalize_slang_dependency(${THIRDPARTY_DIR})
endif()

# Extra Cmake
set(EXTRA_CMAKE ${SAMPLE_FOLDER}/extra.cmake)

if(EXISTS ${EXTRA_CMAKE})
    include(${EXTRA_CMAKE})
    target_sources(${PROJECT_NAME} PRIVATE ${EXTRA_CMAKE})
endif()

# Copy binary
_finalize_target(${PROJECT_NAME})