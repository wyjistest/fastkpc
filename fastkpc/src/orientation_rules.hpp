#ifndef FASTKPC_ORIENTATION_RULES_HPP
#define FASTKPC_ORIENTATION_RULES_HPP

#include "orientation_matrix.hpp"
#include "orientation_types.hpp"

#include <vector>

struct RuleApplicationCounts {
  int rule1;
  int rule2;
  int rule3;
};

bool check_immor(const std::vector<int>& pdag,
                 int p,
                 int V,
                 const std::vector<int>& S);

int orient_colliders(std::vector<int>* pdag,
                     int p,
                     const std::vector<std::vector<std::vector<int> > >& sepsets,
                     bool solve_confl,
                     const std::vector<int>& unf_vect,
                     std::vector<OrientationEvent>* events);

int apply_rule1(std::vector<int>* pdag,
                int p,
                bool solve_confl,
                const std::vector<int>& unf_vect,
                std::vector<OrientationEvent>* events);

int apply_rule2(std::vector<int>* pdag,
                int p,
                bool solve_confl,
                std::vector<OrientationEvent>* events);

int apply_rule3(std::vector<int>* pdag,
                int p,
                bool solve_confl,
                const std::vector<int>& unf_vect,
                std::vector<OrientationEvent>* events);

RuleApplicationCounts apply_rules_until_converged(
  std::vector<int>* pdag,
  int p,
  const OrientationOptions& options,
  const std::vector<int>& unf_vect,
  std::vector<OrientationEvent>* events);

#endif
