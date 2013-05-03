#ifndef _POLPF_WINNER_H_
#define _POLPF_WINNER_H_

#include <vector>
#include <deque>
#include <type_traits>

namespace policy_pf {
namespace winner_policies {

template<typename State, typename Weight, typename Enable = void>
class WeightedArithmeticMean_ {
protected:
	State winner(const std::vector<State>& state_v, const std::vector<Weight>& weight_v) {
		State win = {0};
		for(size_t i = 0; i < state_v.size(); ++i)
			win = win + (state_v[i]*weight_v[i]);
		return win;
	}
};

// Specialization for array type States. As c++ methods cannot return arrays
// winner returns a pointer to a State array.
template<typename State, typename Weight>
class WeightedArithmeticMean_<State, Weight, typename std::enable_if<std::is_array<State>::value >::type> {
protected:
	State* winner(const std::vector<State>& state_v, const std::vector<Weight>& weight_v) {
		State* win = (State*) new State;
		for(size_t j = 0; j < std::extent<State>::value; ++j) {
			(*win)[j] = 0;
			for(size_t i = 0; i < state_v.size(); ++i)
				(*win)[j] = (*win)[j] + (state_v[i][j] * weight_v[i]);
		}
		return win;
	}
};

template <typename T> struct isContainer {static const bool value = false;};
template <typename T> struct isContainer<std::vector<T> > {static const bool value = true;};
template <typename T> struct isContainer<std::deque<T> > {static const bool value = true;};

template<typename State, typename Weight>
class WeightedArithmeticMean_<State, Weight, typename std::enable_if<isContainer<State>::value >::type> {
protected:
	State winner(const std::vector<State>& state_v, const std::vector<Weight>& weight_v) {
		size_t sz = (state_v.size() > 0 ? state_v[0].size() : 0);
	
		State win(sz, 0);
		for(size_t i = 0; i < sz; ++i)
			for(size_t j = 0; j < state_v.size(); ++j)
				win[i] = win[i] + (state_v[j][i]*weight_v[j]);
		return win;
	}
};

#if GCC_VERSION >= 40700
// Workaround for g++ 4.7
template<typename State, typename Weight>
using WeightedArithmeticMean = WeightedArithmeticMean_<State, Weight>;
#else
// Workaround for g++ 4.6
template<typename State, typename Weight>
class WeightedArithmeticMean : public WeightedArithmeticMean_<State, Weight> {};
#endif

}}
#endif
