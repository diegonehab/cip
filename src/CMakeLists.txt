# Copyright 2012--2020 Leonardo Sacht, Rodolfo Schulz de Lima, Diego Nehab
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not
# use this file except in compliance with the License. You may obtain a copy of
# the License at http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
# License for the specific language governing permissions and limitations under
# the License.

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

