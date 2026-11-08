#pragma once
#if defined(__x86_64__) || defined(_M_X64)
#if defined(__GNUC__) || defined(__clang__)
#include_next <emmintrin.h>
#else
// MSVC has no #include_next: its own <emmintrin.h> is the real header, found because this folder is searched after
// the system folders
#include <emmintrin.h>
#endif
#else
#include "immintrin.h"
#endif
