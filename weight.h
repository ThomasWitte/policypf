#ifndef _POLPF_WEIGHT_H_
#define _POLPF_WEIGHT_H_

#include <vector>
#include <type_traits>
#include <cmath>

namespace policy_pf {
namespace weight_policies {

template<typename Weight, typename Observation, typename Enable = void>
class SquareError_ {
protected:
	std::vector<Weight> weight(const std::vector<Observation>& state_v, const Observation& obs) {
		std::vector<Weight> weight_v(state_v.size(), 0);
		for(size_t i = 0; i < state_v.size(); ++i)
			weight_v[i] = 1.0 / ((state_v[i]-obs) * (state_v[i]-obs));
		return weight_v;
	}
};

template<typename Weight, typename Observation>
class SquareError_<Weight, Observation, typename std::enable_if<std::is_array<Observation>::value >::type> {
protected:
	std::vector<Weight> weight(const std::vector<Observation>& state_v, const Observation& obs) {
		std::vector<Weight> weight_v(state_v.size(), 0);
		for(size_t i = 0; i < state_v.size(); ++i)
			for(size_t j = 0; j < std::extent<Observation>::value; ++j)
				weight_v[i] += 1.0 / ((state_v[i][j]-obs[j]) * (state_v[i][j]-obs[j]));
		return weight_v;
	}
};

class NormPdfBase {
public:
	NormPdfBase() : sigma(1), mu(0) {}

	void setNormPdfSigma(double sigma) {
		this->sigma = sigma;
	}

	void setNormPdfMu(double mu) {
		this->mu = mu;
	}

protected:
	double sigma, mu;
};

template<typename Weight, typename Observation, typename Enable = void>
class NormPdf_ : public NormPdfBase {
protected:
	std::vector<Weight> weight(const std::vector<Observation>& state_v, const Observation& obs) {
		std::vector<Weight> weight_v(state_v.size(), 0);
		for(size_t i = 0; i < state_v.size(); ++i) {
			auto x = (state_v[i]-obs);
			weight_v[i] = 1.0/(sigma * sqrt(2*M_PI))
				* pow(M_E, -((x-mu)*(x-mu))/(2.0*sigma*sigma));
		}
		return weight_v;
	}
};

template<typename Weight, typename Observation>
class NormPdf_<Weight, Observation, typename std::enable_if<std::is_array<Observation>::value >::type>
	: public NormPdfBase{
protected:
	std::vector<Weight> weight(const std::vector<Observation>& state_v, const Observation& obs) {
		std::vector<Weight> weight_v(state_v.size(), 0);
		for(size_t i = 0; i < state_v.size(); ++i)
			for(size_t j = 0; j < std::extent<Observation>::value; ++j) {
				auto x = (state_v[i][j]-obs[j]);
				weight_v[i] += 1.0/(sigma * sqrt(2*M_PI))
					* pow(M_E, -((x-mu)*(x-mu))/(2.0*sigma*sigma));
			}
		return weight_v;
	}
};

#if GCC_VERSION >= 40700
// Workaround for g++ 4.7
template<typename Weight, typename Observation>
using SquareError = SquareError_<Weight, Observation>;

template<typename Weight, typename Observation>
using NormPdf = NormPdf_<Weight, Observation>;
#else
// Workaround for g++ 4.6
template<typename Weight, typename Observation>
class SquareError : public SquareError_<Weight, Observation> {};

template<typename Weight, typename Observation>
class NormPdf : public NormPdf_<Weight, Observation> {};
#endif

}}
#endif
