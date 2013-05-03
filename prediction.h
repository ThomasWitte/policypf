#ifndef _POLPF_PREDICTION_H_
#define _POLPF_PREDICTION_H_

#include <vector>

namespace policy_pf {
namespace prediction_policies {

template<typename State>
class None {
protected:
	void predict(std::vector<State> &state_v) {
	}
};

}}

#endif
