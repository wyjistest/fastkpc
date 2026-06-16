#include "orientation_matrix.hpp"

#include <stdexcept>

namespace {

void validate_pdag(const std::vector<int>& pdag, int p) {
  if (p < 0 || static_cast<int>(pdag.size()) != p * p) {
    throw std::runtime_error("pdag dimension mismatch");
  }
}

void validate_edge_value(int value) {
  if (value != FASTKPC_EDGE_NONE &&
      value != FASTKPC_EDGE_PRESENT &&
      value != FASTKPC_EDGE_CONFLICT) {
    throw std::runtime_error("invalid pdag edge value");
  }
}

}  // namespace

int pdag_index(int p, int row, int col) {
  if (p < 0 || row < 0 || col < 0 || row >= p || col >= p) {
    throw std::runtime_error("pdag index out of range");
  }
  return row * p + col;
}

int pdag_get(const std::vector<int>& pdag, int p, int row, int col) {
  validate_pdag(pdag, p);
  return pdag[static_cast<std::size_t>(pdag_index(p, row, col))];
}

void pdag_set(std::vector<int>* pdag, int p, int row, int col, int value) {
  validate_pdag(*pdag, p);
  validate_edge_value(value);
  (*pdag)[static_cast<std::size_t>(pdag_index(p, row, col))] = value;
}

bool has_any_edge(const std::vector<int>& pdag, int p, int a, int b) {
  return pdag_get(pdag, p, a, b) != FASTKPC_EDGE_NONE ||
    pdag_get(pdag, p, b, a) != FASTKPC_EDGE_NONE;
}

bool has_undirected_edge(const std::vector<int>& pdag, int p, int a, int b) {
  return pdag_get(pdag, p, a, b) == FASTKPC_EDGE_PRESENT &&
    pdag_get(pdag, p, b, a) == FASTKPC_EDGE_PRESENT;
}

bool has_directed_edge(const std::vector<int>& pdag, int p, int a, int b) {
  return pdag_get(pdag, p, a, b) == FASTKPC_EDGE_PRESENT &&
    pdag_get(pdag, p, b, a) == FASTKPC_EDGE_NONE;
}

bool has_conflict_edge(const std::vector<int>& pdag, int p, int a, int b) {
  return pdag_get(pdag, p, a, b) == FASTKPC_EDGE_CONFLICT &&
    pdag_get(pdag, p, b, a) == FASTKPC_EDGE_CONFLICT;
}

void set_no_edge(std::vector<int>* pdag, int p, int a, int b) {
  pdag_set(pdag, p, a, b, FASTKPC_EDGE_NONE);
  pdag_set(pdag, p, b, a, FASTKPC_EDGE_NONE);
}

void set_undirected_edge(std::vector<int>* pdag, int p, int a, int b) {
  pdag_set(pdag, p, a, b, FASTKPC_EDGE_PRESENT);
  pdag_set(pdag, p, b, a, FASTKPC_EDGE_PRESENT);
}

void set_directed_edge(std::vector<int>* pdag, int p, int from, int to) {
  pdag_set(pdag, p, from, to, FASTKPC_EDGE_PRESENT);
  pdag_set(pdag, p, to, from, FASTKPC_EDGE_NONE);
}

void set_conflict_edge(std::vector<int>* pdag, int p, int a, int b) {
  pdag_set(pdag, p, a, b, FASTKPC_EDGE_CONFLICT);
  pdag_set(pdag, p, b, a, FASTKPC_EDGE_CONFLICT);
}

std::vector<int> pdag_from_skeleton_adjacency(const std::vector<int>& adjacency, int p) {
  if (p < 0 || static_cast<int>(adjacency.size()) != p * p) {
    throw std::runtime_error("pdag dimension mismatch");
  }
  std::vector<int> pdag(static_cast<std::size_t>(p) * p, FASTKPC_EDGE_NONE);
  for (int row = 0; row < p; ++row) {
    for (int col = 0; col < p; ++col) {
      if (row != col &&
          adjacency[static_cast<std::size_t>(row) * p + col] != 0) {
        pdag[static_cast<std::size_t>(row) * p + col] = FASTKPC_EDGE_PRESENT;
      }
    }
  }
  return pdag;
}
