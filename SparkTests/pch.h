//
// pch.h
// Header for standard system include files.
//

#pragma once

#include "gtest/gtest.h"

#include <rttr/registration>

#define GCOUT std::cerr << "[INFO] "
inline void gprintf(_In_z_ _Printf_format_string_ char const* const _Format, ...) {
	fprintf(stderr, "[INFO] ");
	va_list _ArgList;
	__crt_va_start(_ArgList, _Format);
	fprintf(stderr, _Format, _ArgList);
	__crt_va_end(_ArgList);
}