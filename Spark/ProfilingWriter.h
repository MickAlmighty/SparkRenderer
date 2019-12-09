#ifndef PROFILING_WRITER_H
#define PROFILING_WRITER_H

#include <string>
#include <fstream>

#include "Structs.h"
#include <mutex>

namespace spark {


	class ProfilingWriter
	{
	public:
		ProfilingWriter();

		void beginSession(const std::string& name, const std::string& filepath = "profilingCapture.json");
		void endSession();
		void writeRecord(const spark::ProfileRecord& result);

		static ProfilingWriter& get();

	private:
		//InstrumentationSession* m_CurrentSession;
		std::ofstream outputStream;
		int profileCount;
		std::mutex ofstreamMutex;

		void writeHeader();
		void writeFooter();
	};
}
#endif