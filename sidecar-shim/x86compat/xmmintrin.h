#pragma once
#if defined(__x86_64__) || defined(_M_X64)
#if defined(_MSC_VER)
#include <xmmintrin.h>
#else
#include_next <xmmintrin.h>
#endif
#else
#include "immintrin.h"
#endif
