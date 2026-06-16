#include "orientation_rules.hpp"

#include <algorithm>
#include <limits>
#include <string>

namespace {

bool contains_node(const std::vector<int>& values, int node) {
  return std::find(values.begin(), values.end(), node) != values.end();
}

bool edge_state_equals(const std::vector<int>& pdag,
                       int p,
                       int a,
                       int b,
                       int ab,
                       int ba) {
  return pdag_get(pdag, p, a, b) == ab && pdag_get(pdag, p, b, a) == ba;
}

OrientationEvent make_event(const std::string& phase,
                            const std::string& rule,
                            int x,
                            int y,
                            int z,
                            const std::vector<int>& S,
                            double p_value,
                            bool accepted,
                            const std::string& message) {
  OrientationEvent event;
  event.phase = phase;
  event.rule = rule;
  event.x = x;
  event.y = y;
  event.z = z;
  event.S = S;
  event.p_value = p_value;
  event.accepted = accepted;
  event.message = message;
  return event;
}

void add_unf_diagnostic(const std::vector<int>& unf_vect,
                        std::vector<OrientationEvent>* events,
                        const std::string& rule) {
  if (unf_vect.empty() || events == NULL) return;
  events->push_back(make_event("rules", rule, -1, -1, -1, std::vector<int>(),
                               std::numeric_limits<double>::quiet_NaN(), false,
                               "unfVect not implemented in native orientation"));
}

bool orient_edge(std::vector<int>* pdag,
                 int p,
                 int from,
                 int to,
                 bool solve_confl) {
  const int old_from_to = pdag_get(*pdag, p, from, to);
  const int old_to_from = pdag_get(*pdag, p, to, from);
  if (!solve_confl) {
    set_directed_edge(pdag, p, from, to);
  } else if (old_from_to == FASTKPC_EDGE_PRESENT) {
    pdag_set(pdag, p, to, from, FASTKPC_EDGE_NONE);
  } else {
    set_conflict_edge(pdag, p, from, to);
  }
  return !edge_state_equals(*pdag, p, from, to, old_from_to, old_to_from);
}

bool orient_rule_edge(std::vector<int>* pdag,
                      int p,
                      int from,
                      int to,
                      bool solve_confl) {
  const int old_from_to = pdag_get(*pdag, p, from, to);
  const int old_to_from = pdag_get(*pdag, p, to, from);
  if (!solve_confl || has_undirected_edge(*pdag, p, from, to)) {
    set_directed_edge(pdag, p, from, to);
  } else if (has_directed_edge(*pdag, p, to, from)) {
    set_conflict_edge(pdag, p, from, to);
  }
  return !edge_state_equals(*pdag, p, from, to, old_from_to, old_to_from);
}

std::vector<std::pair<int, int> > directed_pairs(const std::vector<int>& pdag,
                                                 int p) {
  std::vector<std::pair<int, int> > pairs;
  for (int a = 0; a < p; ++a) {
    for (int b = 0; b < p; ++b) {
      if (a != b && has_directed_edge(pdag, p, a, b)) {
        pairs.push_back(std::make_pair(a, b));
      }
    }
  }
  return pairs;
}

std::vector<std::pair<int, int> > undirected_scan_pairs(
    const std::vector<int>& pdag,
    int p) {
  std::vector<std::pair<int, int> > pairs;
  for (int a = 0; a < p; ++a) {
    for (int b = 0; b < p; ++b) {
      if (a != b && has_undirected_edge(pdag, p, a, b)) {
        pairs.push_back(std::make_pair(a, b));
      }
    }
  }
  return pairs;
}

bool sepset_contains(const std::vector<std::vector<std::vector<int> > >& sepsets,
                     int a,
                     int b,
                     int node) {
  if (a < 0 || b < 0 ||
      a >= static_cast<int>(sepsets.size()) ||
      b >= static_cast<int>(sepsets[a].size())) {
    return false;
  }
  return contains_node(sepsets[a][b], node);
}

}  // namespace

bool check_immor(const std::vector<int>& pdag,
                 int p,
                 int V,
                 const std::vector<int>& S) {
  for (std::size_t i = 0; i < S.size(); ++i) {
    const int a = S[i];
    if (a < 0 || a >= p || a == V) return false;
    for (std::size_t j = i + 1; j < S.size(); ++j) {
      const int b = S[j];
      if (b < 0 || b >= p || b == V) return false;
      if (!has_any_edge(pdag, p, a, b)) return false;
    }
  }

  std::vector<int> parents;
  for (int node = 0; node < p; ++node) {
    if (node != V && has_directed_edge(pdag, p, node, V)) {
      parents.push_back(node);
    }
  }

  for (int parent : parents) {
    for (int node : S) {
      if (!has_any_edge(pdag, p, parent, node)) return false;
    }
  }
  return true;
}

int orient_colliders(std::vector<int>* pdag,
                     int p,
                     const std::vector<std::vector<std::vector<int> > >& sepsets,
                     bool solve_confl,
                     const std::vector<int>& unf_vect,
                     std::vector<OrientationEvent>* events) {
  add_unf_diagnostic(unf_vect, events, "collider");
  int count = 0;
  const std::vector<int> initial = *pdag;
  for (int x = 0; x < p; ++x) {
    for (int y = 0; y < p; ++y) {
      if (x == y || pdag_get(initial, p, x, y) == FASTKPC_EDGE_NONE) continue;
      for (int z = 0; z < p; ++z) {
        if (z == x || z == y) continue;
        if (pdag_get(initial, p, y, z) == FASTKPC_EDGE_NONE) continue;
        if (has_any_edge(initial, p, x, z)) continue;
        if (sepset_contains(sepsets, x, z, y) ||
            sepset_contains(sepsets, z, x, y)) {
          continue;
        }

        const bool changed_xy = orient_edge(pdag, p, x, y, solve_confl);
        const bool changed_zy = orient_edge(pdag, p, z, y, solve_confl);
        if (changed_xy || changed_zy) {
          ++count;
          if (events != NULL) {
            std::vector<int> S;
            events->push_back(make_event("collider", "collider", x, y, z, S,
                                         std::numeric_limits<double>::quiet_NaN(),
                                         true, "oriented collider"));
          }
        }
      }
    }
  }
  return count;
}

int apply_rule1(std::vector<int>* pdag,
                int p,
                bool solve_confl,
                const std::vector<int>& unf_vect,
                std::vector<OrientationEvent>* events) {
  add_unf_diagnostic(unf_vect, events, "rule1");
  int count = 0;
  const std::vector<std::pair<int, int> > pairs = directed_pairs(*pdag, p);
  for (std::size_t i = 0; i < pairs.size(); ++i) {
    const int a = pairs[i].first;
    const int b = pairs[i].second;
    for (int c = 0; c < p; ++c) {
      if (c == a || c == b) continue;
      if (!has_undirected_edge(*pdag, p, b, c)) continue;
      if (has_any_edge(*pdag, p, a, c)) continue;
      if (orient_rule_edge(pdag, p, b, c, solve_confl)) {
        ++count;
        if (events != NULL) {
          std::vector<int> S;
          events->push_back(make_event("rules", "rule1", a, b, c, S,
                                       std::numeric_limits<double>::quiet_NaN(),
                                       true, "oriented by rule1"));
        }
      }
    }
  }
  return count;
}

int apply_rule2(std::vector<int>* pdag,
                int p,
                bool solve_confl,
                std::vector<OrientationEvent>* events) {
  int count = 0;
  const std::vector<std::pair<int, int> > pairs = undirected_scan_pairs(*pdag, p);
  for (std::size_t i = 0; i < pairs.size(); ++i) {
    const int a = pairs[i].first;
    const int b = pairs[i].second;
    if (!has_undirected_edge(*pdag, p, a, b)) continue;
    for (int c = 0; c < p; ++c) {
      if (c == a || c == b) continue;
      if (!has_directed_edge(*pdag, p, a, c)) continue;
      if (!has_directed_edge(*pdag, p, c, b)) continue;
      if (orient_rule_edge(pdag, p, a, b, solve_confl)) {
        ++count;
        if (events != NULL) {
          std::vector<int> S;
          events->push_back(make_event("rules", "rule2", a, b, c, S,
                                       std::numeric_limits<double>::quiet_NaN(),
                                       true, "oriented by rule2"));
        }
      }
      break;
    }
  }
  return count;
}

int apply_rule3(std::vector<int>* pdag,
                int p,
                bool solve_confl,
                const std::vector<int>& unf_vect,
                std::vector<OrientationEvent>* events) {
  add_unf_diagnostic(unf_vect, events, "rule3");
  int count = 0;
  const std::vector<std::pair<int, int> > pairs = undirected_scan_pairs(*pdag, p);
  for (std::size_t i = 0; i < pairs.size(); ++i) {
    const int a = pairs[i].first;
    const int b = pairs[i].second;
    if (!has_undirected_edge(*pdag, p, a, b)) continue;

    std::vector<int> candidates;
    for (int c = 0; c < p; ++c) {
      if (c == a || c == b) continue;
      if (has_undirected_edge(*pdag, p, a, c) &&
          has_directed_edge(*pdag, p, c, b)) {
        candidates.push_back(c);
      }
    }

    bool oriented = false;
    for (std::size_t c1_index = 0; c1_index < candidates.size() && !oriented;
         ++c1_index) {
      for (std::size_t c2_index = c1_index + 1; c2_index < candidates.size();
           ++c2_index) {
        const int c1 = candidates[c1_index];
        const int c2 = candidates[c2_index];
        if (has_any_edge(*pdag, p, c1, c2)) continue;
        if (orient_rule_edge(pdag, p, a, b, solve_confl)) {
          ++count;
          if (events != NULL) {
            std::vector<int> S;
            S.push_back(c1);
            S.push_back(c2);
            events->push_back(make_event("rules", "rule3", a, b, -1, S,
                                         std::numeric_limits<double>::quiet_NaN(),
                                         true, "oriented by rule3"));
          }
        }
        oriented = true;
        break;
      }
    }
  }
  return count;
}

RuleApplicationCounts apply_rules_until_converged(
  std::vector<int>* pdag,
  int p,
  const OrientationOptions& options,
  const std::vector<int>& unf_vect,
  std::vector<OrientationEvent>* events) {
  RuleApplicationCounts total;
  total.rule1 = 0;
  total.rule2 = 0;
  total.rule3 = 0;

  bool changed = true;
  while (changed) {
    const std::vector<int> before = *pdag;
    if (options.rule1) {
      total.rule1 += apply_rule1(pdag, p, options.solve_confl, unf_vect, events);
    }
    if (options.rule2) {
      total.rule2 += apply_rule2(pdag, p, options.solve_confl, events);
    }
    if (options.rule3) {
      total.rule3 += apply_rule3(pdag, p, options.solve_confl, unf_vect, events);
    }
    changed = (*pdag != before);
  }
  return total;
}
