#ifndef PARTICLE_FILTER_H_
#define PARTICLE_FILTER_H_

#include <vector>
#include <type_traits>

#define GCC_VERSION (__GNUC__ * 10000 \
					+ __GNUC_MINOR__ * 100 \
					+ __GNUC_PATCHLEVEL__)

#include "resampling.h"
#include "weight.h"
#include "noise.h"
#include "init.h"
#include "prediction.h"
#include "winner.h"
#include "state2obs.h"

namespace policy_pf {

/*
 * this particle filter template creates a particle filter
 * using the provided policies and types.
 */

template<

	// StateType is the type of the state the system is in (default double)
	typename StateType = double,

	// ObservationType is the type of the observation the particle filter
	// receives (default double)
	typename ObservationType = double,

	// WeightType is the type the particle filter uses to measure the
	// probability of a state (default double)
	typename WeightType = double,

	// PredictionPolicy defines the method predict that applies the system equation
	// to a vector of states/particles
	template<class> class PredictionPolicy
		= prediction_policies::None,

	// State2Obs defines the method state2obs that applies the observation equation
	// to a vector of states returning the corresponding observations
	template<class, class> class State2Obs
		= state2obs::Identity,

	// WeightPolicy defines the method weight that returns a probability vector
	// given a vector of observations and a reference observation
	template<class, class> class WeightPolicy
		= weight_policies::NormPdf,

	// WinnerPolicy defines the method winner that returns a winning state based
	// on the state and probability vectors
	template<class, class> class WinnerPolicy
		= winner_policies::WeightedArithmeticMean,

	// InitPolicy defines the method init that initializes the state vector
	// when the particle filter starts
	template<class> class InitPolicy
		= init_policies::Gaussian,

	// NoisePolicy defines the method noise that adds systen noise to the state vector
	template<class> class NoisePolicy
		= noise_policies::GaussianNoise,

	// ResamplingPolicy defines the method resampling that performs a resampling
	// step on the state and weight vectors
	template<class, class> class ResamplingPolicy
		= resampling_policies::SystematicResampling>

class ParticleFilter :
	public PredictionPolicy<StateType>,
	public WeightPolicy<WeightType, ObservationType>,
	public ResamplingPolicy<StateType, WeightType>,
	public NoisePolicy<StateType>,
	public InitPolicy<StateType>,
	public WinnerPolicy<StateType, WeightType>,
	public State2Obs<StateType, ObservationType>
{
public:
	ParticleFilter(unsigned int num_particles) :
		PredictionPolicy<StateType>(),
		WeightPolicy<WeightType, ObservationType>(),
		ResamplingPolicy<StateType, WeightType>(),
		NoisePolicy<StateType>(),
		InitPolicy<StateType>(),
		WinnerPolicy<StateType, WeightType>(),
		State2Obs<StateType, ObservationType>(),
		num_particles(num_particles), initialized(false) {}

	~ParticleFilter() {}

#if GCC_VERSION < 40700
	typedef ParticleFilter<StateType, ObservationType, WeightType, PredictionPolicy,
		State2Obs, WeightPolicy, WinnerPolicy, InitPolicy, NoisePolicy, ResamplingPolicy> PF_t;
	static PF_t *getPFInstance();
#define THIS getPFInstance()
#else
#define THIS this
#endif

	auto run(ObservationType& observation)
			-> decltype(THIS->WinnerPolicy<StateType, WeightType>::winner(
				std::vector<StateType>(), std::vector<WeightType>())) {
		static_assert(std::is_floating_point<WeightType>::value,
			"WeightType must be a floating point type!");
	
		// initialize the state vector
		if(!initialized) {
			particles = InitPolicy<StateType>::init(num_particles);
			initialized = true;
		}

		// update particles and add system noise
		PredictionPolicy<StateType>::predict(particles);
		NoisePolicy<StateType>::noise(particles);

		// calculate weights/probabilities
		auto hyp_obs = State2Obs<StateType, ObservationType>::state2obs(particles);
		auto particle_weights = WeightPolicy<WeightType, ObservationType>::weight(hyp_obs, observation);

		// normalize weights
		WeightType wsum = 0;
		for(auto w : particle_weights) {
			wsum += w;
		}
		for(size_t i = 0; i < num_particles; ++i) {
			particle_weights[i] /= wsum;
		}

		// resampling
		ResamplingPolicy<StateType, WeightType>::resampling(particles, particle_weights);

		// choose winner
		return WinnerPolicy<StateType, WeightType>::winner(particles, particle_weights);
	}

	void reset() {
		initialized = false;
	}

private:
	unsigned int num_particles;
	bool initialized;
	std::vector<StateType> particles;
};

}

#undef THIS
#undef GCC_VERSION

#endif
