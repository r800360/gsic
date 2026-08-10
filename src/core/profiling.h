#pragma once

// Tracy instrumentation, compiled out entirely unless GSIC_ENABLE_TRACY is set.
#if defined(GSIC_ENABLE_TRACY)
  #include <tracy/Tracy.hpp>
  #define GSIC_ZONE(name) ZoneScopedN(name)
  #define GSIC_FRAME() FrameMark
#else
  #define GSIC_ZONE(name)
  #define GSIC_FRAME()
#endif
