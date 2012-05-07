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

