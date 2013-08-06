#ifndef _POLPF_RESAMPLING_H_
#define _POLPF_RESAMPLING_H_

#include <random>
#include <vector>
#include <numeric>

namespace policy_pf {
namespace resampling_policies {

template<typename State, typename Weight>
class SystematicResampling {
protected:
	void resampling(std::vector<State> &state_v, std::vector<Weight> &weight_v) {
	
		unsigned int num_particles = weight_v.size();
		/*
		 * Es wird die Kumulative Summe der (normalisierten) Gewichte berechnet, sodass man
		 * sich edges als Treppe mit je nach Gewicht unterschiedlich hohen Stufen vorstellen kann.
		 */
		std::vector<Weight> edges(num_particles + 1, 0);
        std::partial_sum(weight_v.cbegin(), weight_v.cend(), edges.begin()+1);
        for(auto& e : edges) e > 1 ? e = 1 : 0;
        edges.back() = 1;

		/*
		 * Ein Startwert [0,1/n] wird zufällig gewählt und danach jeweils um 1/n erhöht.
		 * Dabei wird jedes mal verglichen, welche Stufe derzeit erreicht wird und das ensprechende
		 * Partikel aus dem Partikelvektor übernommen.
		 * Dadurch wird sichergestellt, dass jedes Partikel proportional häufig zu seinem Gewicht
		 * ausgewählt wird.
		 */
		Weight u1 = uni_rand(generator) / num_particles;
		std::vector<State> new_state_v;

		for(size_t i = 1; i < edges.size(); ++i) {
			while(u1 < edges[i]) {
				new_state_v.push_back(state_v[i-1]);
				u1 += 1.0 / num_particles;
			}
			weight_v[i-1] = 1.0 / num_particles;
		}

		state_v = new_state_v;
	}
	
private:
	std::default_random_engine generator;
	std::uniform_real_distribution<Weight> uni_rand;
};

}}

#endif
