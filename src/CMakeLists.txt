include_directories(${FLTK_INCLUDE_DIR}
    ${OPENGL_INCLUDE_DIR} ${GLEW_INCLUDE_DIRS})

set(CUDA_BUILD_CUBIN off)

set(HAS_GLEW ${GLEW_FOUND})

configure_file(config.h.in config.h ESCAPE_QUOTES)

fltk_wrap_ui(nlfilter nlfilter_gui.fl)

set(SOURCES nlfilter.cpp image.cpp recfilter.cu image_frame.cpp 
            threads.cpp
    filter.cu vsync.cpp timer.cpp image_util.cu
    ${nlfilter_FLTK_UI_SRCS})

cuda_add_executable(nlfilter ${SOURCES})
target_link_libraries(nlfilter ${FLTK_LIBRARIES}
    ${OPENGL_LIBRARIES} ${GLEW_LIBRARIES} pthread)

