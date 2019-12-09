#include "ProfilingWriter.h"


spark::ProfilingWriter::ProfilingWriter(): profileCount(0)
{
}

void spark::ProfilingWriter::beginSession(const std::string& name, const std::string& filepath)
{
	outputStream.open(filepath);
	writeHeader();
	//m_CurrentSession = new InstrumentationSession{ name };
}

void spark::ProfilingWriter::endSession()
{
	writeFooter();
	outputStream.close();
	//delete m_CurrentSession;
	//m_CurrentSession = nullptr;
	profileCount = 0;
}

void spark::ProfilingWriter::writeRecord(const spark::ProfileRecord& result)
{
	std::lock_guard<std::mutex> lock(ofstreamMutex);
	if (profileCount++ > 0)
		outputStream << ",";

	std::string name = result.name;
	std::replace(name.begin(), name.end(), '"', '\'');

	outputStream << "{"
	<< "\"cat\":\"function\","
	<< "\"dur\":" << (result.end - result.start) << ','
	<< "\"name\":\"" << name << "\","
	<< "\"ph\":\"X\","
	<< "\"pid\":0,"
	<< "\"tid\":" << result.ThreadID << ","
	<< "\"ts\":" << result.start
	<< "}";

	outputStream.flush();
}

void spark::ProfilingWriter::writeHeader()
{
	outputStream << "{\"otherData\": {},\"traceEvents\":[";
	outputStream.flush();
}

void spark::ProfilingWriter::writeFooter()
{
	outputStream << "]}";
	outputStream.flush();
}

spark::ProfilingWriter& spark::ProfilingWriter::get()
{
	static ProfilingWriter instance;
	return instance;
}
