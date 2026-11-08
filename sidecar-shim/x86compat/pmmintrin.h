#pragma once
#if (defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)) && (defined(__GNUC__) || defined(__clang__))
#include_next <pmmintrin.h>
#else
#include "immintrin.h"
#endif
