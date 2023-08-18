# slang
# set(SLANG_DIR ${CMAKE_CURRENT_SOURCE_DIR}/slang)
# add_library(slang SHARED IMPORTED GLOBAL)
# set_target_properties(slang PROPERTIES
# INTERFACE_INCLUDE_DIRECTORIES ${SLANG_DIR}
# IMPORTED_IMPLIB ${SLANG_DIR}/bin/windows-x64/release/slang.lib
# IMPORTED_LOCATION ${SLANG_DIR}/bin/windows-x64/release/slang.dll
# )
# file(GLOB SLANG_BIN LIST_DIRECTORIES false "${SLANG_DIR}/bin/windows-x64/release/*.dll" "${SLANG_DIR}/bin/windows-x64/release/*.lib")
# message("slang dir ${SLANG_DIR}")

function(configure_third_party_libs target)
    target_link_libraries(${target} sdl2 VulkanMemoryAllocator slang)
endfunction()

function(finalize_slang_dependency thirdparty_root)
    message("-- copy slang dependencies")
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

function(finalize_dxc_dependency thirdparty_root)
    message("-- copy slang dependencies")
    set(DXC_DIR ${thirdparty_root}/dxc)
    file(GLOB DXC_BIN LIST_DIRECTORIES false "${DXC_DIR}/bin/*")

    foreach(bin ${DXC_BIN})
        add_custom_command(
            TARGET ${PROJECT_NAME} POST_BUILD
            COMMAND ${CMAKE_COMMAND} ARGS -E copy ${bin} $<TARGET_FILE_DIR:${PROJECT_NAME}>
            COMMENT "Copy binary to $<TARGET_FILE_DIR:${PROJECT_NAME}>"
        )
    endforeach()
endfunction()