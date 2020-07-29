// Copyright 2012--2020 Leonardo Sacht, Rodolfo Schulz de Lima, Diego Nehab
//
// Licensed under the Apache License, Version 2.0 (the "License"); you may not
// use this file except in compliance with the License. You may obtain a copy of
// the License at http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
// WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
// License for the specific language governing permissions and limitations under
// the License.

#include "config.h"

#if HAS_GLEW
#   include <GL/glew.h>
#   ifdef _WIN32
#       include <GL/wglew.h>
#   else
#       include <GL/glxew.h>
#   endif
#endif

bool g_vsync_enabled = false;

bool use_vsync()
{
#if HAS_GLEW
#   if defined(_WIN32)
    return wglSwapIntervalEXT?true:false;
#   else
    return glXSwapIntervalSGI?true:false;
#   endif
#endif
    return false;
}

void enable_vsync(bool en)
{
    if(!use_vsync() || g_vsync_enabled == en)
        return;

#if HAS_GLEW
#   if defined(_WIN32)
    wglSwapIntervalEXT(en);
#   else
    glXSwapIntervalSGI(en);
#   endif
    g_vsync_enabled = en;
#endif
}

