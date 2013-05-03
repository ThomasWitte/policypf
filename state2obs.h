#ifndef _POLPF_STATE2OBS_H_
#define _POLPF_STATE2OBS_H_

#include <vector>

namespace policy_pf {
namespace state2obs {

template <typename State, typename Observation>
class Identity {
protected:
	std::vector<State> state2obs(const std::vector<Observation>& obs) {
		return obs;
	}
};

}}

#endif
