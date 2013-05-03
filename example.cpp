#include <vector>
#include <cmath>
#include <iostream>
#include <random>
#include "ParticleFilter.h"

// observation and process equation

double generate_observation(double state) {
	return state*state / 20.0;
}

double system(double old, int k) {
	return old/2.0 + 25*old/(1+old*old) + 8*cos(1.2*k);
}

// wrapper around the process equation, that is used by the
// particle filter

template <typename State>
class ExPrediction {
protected:
	ExPrediction() : k(0) {}

	void predict(std::vector<State>& state_v) {
		k++;
		for(size_t i = 0; i < state_v.size(); ++i)
			state_v[i] = system(state_v[i], k);
	}

private:
	int k;

};

// wrapper around the observation equation

template <typename State, typename Obs>
class ExState2Obs {
protected:
	std::vector<Obs> state2obs(std::vector<State>& state_v) {
		std::vector<Obs> obs_v(state_v.size(), 0);
		for(size_t i = 0; i < state_v.size(); ++i)
			obs_v[i] = generate_observation(state_v[i]);
		return obs_v;
	}
};

int main(int argc, char *argv[]) {
	// random generators for the simulated process
	std::random_device gen;
	std::normal_distribution<double> obs_noise(0,1);
	std::normal_distribution<double> sys_noise(0,10);

	// define a particle filter that uses our custom
	// observation and process equations
	typedef policy_pf::ParticleFilter<
		double, // state type
		double, // observation type
		double, // weight type
		ExPrediction, // process equation
		ExState2Obs>  // observation equation
	MyParticleFilter;
	
	// create a particle filter instance with 1000 particles
	MyParticleFilter mpf(1000);

	// set sigma for initial and process noise
	mpf.setInitSigma(sqrt(10));
	mpf.setNoiseSigma(sqrt(10));

	double sys = 0; // initial system state
	double obs = 0; // initial observation
	double est_sys = 0; // initial estimated system state
	double est_obs = 0; // initial esimated observation
	for(int i = 1; i <= 40; ++i) {
		// simulate the system
		sys = system(sys, i) + sys_noise(gen);
		// generate an observation from the simulated system state
		obs = generate_observation(sys) + obs_noise(gen);
		// estimate the system state using the particle filter
		est_sys = mpf.run(obs);
		// get the filtered observation from the estimated system state
		est_obs = generate_observation(est_sys);

		std::cout << sys << "\t" << obs << "\t" << est_sys << "\t" << est_obs << std::endl;
	}
	return 0;
}
