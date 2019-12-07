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
	if (profileCount++ > 0)
		outputStream << ",";

	std::string name = result.name;
	std::replace(name.begin(), name.end(), '"', '\'');

	outputStream << "{";
	outputStream << "\"cat\":\"function\",";
	outputStream << "\"dur\":" << (result.end - result.start) << ',';
	outputStream << "\"name\":\"" << name << "\",";
	outputStream << "\"ph\":\"X\",";
	outputStream << "\"pid\":0,";
	outputStream << "\"tid\":" << result.ThreadID << ",";
	outputStream << "\"ts\":" << result.start;
	outputStream << "}";

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
