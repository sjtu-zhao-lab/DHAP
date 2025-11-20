#ifndef DISAGG_UTIL_HEADER_H
#define DISAGG_UTIL_HEADER_H

#include <iostream>
#include <streambuf>
#include <string>
#include <mpi.h>
#include <assert.h>

#define INP_MEM_RATIO 0.5

using row_size_t = uint32_t;
#define MPI_ROW_SIZE_T MPI_UINT32_T
using buf_size_t = uint64_t;
#define MPI_BUF_SIZE_T MPI_UINT64_T

#define TIME_COST(stat, code)																				\
	do {																															\
		auto start = std::chrono::high_resolution_clock::now(); 				\
		code;                                      											\
		auto end = std::chrono::high_resolution_clock::now();   				\
		auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start); \
		stat = duration;																								\
	} while(0)

#define TIME_COST_ACC(stat, code)																				\
	do {																															\
		auto start = std::chrono::high_resolution_clock::now(); 				\
		code;                                      											\
		auto end = std::chrono::high_resolution_clock::now();   				\
		auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start); \
		stat += duration;																								\
	} while(0)

#define MPI_CHECK(stmt)                                                        \
	do {                                                                       \
		int mpi_errno = (stmt);                                                \
		if (MPI_SUCCESS != mpi_errno) {                                        \
			fprintf(stderr, "[%s:%d] MPI call failed with %d \n", __FILE__,    \
							__LINE__, mpi_errno);                                      \
			exit(EXIT_FAILURE);                                                \
		}                                                                      \
		assert(MPI_SUCCESS == mpi_errno);                                      \
	} while (0)

// Custom stream buffer that adds a prefix to each line of output
class PrefixBuf : public std::streambuf {
public:
	PrefixBuf(std::streambuf* sbuf, const std::string& prefix)
		: sbuf_(sbuf), prefix_(prefix), need_prefix_(true) {}

protected:
	int_type overflow(int_type c) override {
		if (c == '\n') {
			need_prefix_ = true;  // Add the prefix before the next line
		}
		else if (need_prefix_) {
			// Add the prefix
			for (char prefixChar : prefix_) {
				if (sbuf_->sputc(prefixChar) == traits_type::eof()) {
					return traits_type::eof();
				}
			}
			need_prefix_ = false;  // Prefix added, no longer needed
		}
		return sbuf_->sputc(c);
	}

private:
	std::streambuf* sbuf_;
	std::string prefix_;
	bool need_prefix_;
};

#endif