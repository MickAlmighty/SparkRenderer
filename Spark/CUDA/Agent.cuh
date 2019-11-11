#ifndef AGENT_CUH
#define AGENT_CUH

namespace spark {
	namespace cuda {
		class Agent {
		public:
			int indexBegin { 0 };
			int pathSize { 0 };
			int points[4] = {0, 0, 0, 0};
			int* pathOutput = nullptr;
		};
	}
}
#endif
