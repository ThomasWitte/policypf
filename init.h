#ifndef _POLPF_INIT_H_
#define _POLPF_INIT_H_

#include <random>
#include <vector>
#include <deque>
#include <array>
#include <type_traits>
#include <algorithm>

namespace policy_pf {
namespace init_policies {

// GaussianBase encapsulates a gaussian distributed random number generator
template<typename FloatingType>
class GaussianBase {
public:
	inline void setInitSigma(FloatingType sigma) {
		uni_rand = std::normal_distribution<FloatingType>(0, sigma);
	}

protected:
	inline FloatingType random() {
		return uni_rand(generator);
	}

private:
	std::default_random_engine generator;
	std::normal_distribution<FloatingType> uni_rand;
};

// compilation fails if no type trait is applicable
template<typename State, typename Enable = void>
class AutoDetect;

// implementation for scalar floating point types
template<typename State>
class AutoDetect<State, typename std::enable_if<std::is_floating_point<State>::value >::type>
	: public GaussianBase<State> {
protected:
	inline void apply_init(State& s) {
		s += this->random();
	}
};

// recursive implementation for arrays
template<typename State>
class AutoDetect<State, typename std::enable_if<std::is_array<State>::value >::type>
	: public AutoDetect<typename std::remove_extent<State>::type> {
protected:
	inline void apply_init(State& s) {
		for(size_t i = 0; i < std::extent<State>::value; ++i)
			AutoDetect<typename std::remove_extent<State>::type>::apply_init(s[i]);
	}
};

// recursive implementation for standard container types vector, deque and array
template <typename T> struct isContainer {static const bool value = false;};
template <typename T> struct isContainer<std::vector<T> > {static const bool value = true;};
template <typename T> struct isContainer<std::deque<T> > {static const bool value = true;};
template <typename T, int N> struct isContainer<std::array<T, N> > {static const bool value = true;};

template<typename State>
class AutoDetect<State, typename std::enable_if<isContainer<State>::value >::type>
	: public AutoDetect<typename State::value_type> {
protected:
	inline void apply_init(State& s) {
		for(size_t i = 0; i < s.size(); ++i)
			AutoDetect<typename State::value_type>::apply_init(s[i]);
	}
};

#if GCC_VERSION >= 40700
// Workaround for g++ 4.7
template<typename State> using AutoDetectT = AutoDetect<State>;
#else
// Workaround for g++ 4.6
template<typename State> class AutoDetectT : public AutoDetect<State> {};
#endif

// Appies gaussian noise based on an ApplicationPolicy;
// the default policy AutoDetect works for scalar floating point types,
// (multidimensional) arrays, std::vector, std::deque and std::array
template<typename State, template<class> class ApplicationPolicy = AutoDetectT>
class Gaussian_ : public ApplicationPolicy<State> {
protected:
	std::vector<State> init(unsigned int sz) {
		std::vector<State> res(sz, 0);
		for(size_t i = 0; i < sz; ++i) {
			this->apply_init(res[i]);
		}
		return res;
	}
};

#if GCC_VERSION >= 40700
// Workaround for g++ 4.7
template<typename State>
using Gaussian = Gaussian_<State>;
#else
// Workaround for g++ 4.6
template<typename State>
class Gaussian : public Gaussian_<State> {};
#endif

}}
#endif
