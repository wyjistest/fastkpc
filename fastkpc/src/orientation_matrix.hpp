#ifndef FASTKPC_ORIENTATION_MATRIX_HPP
#define FASTKPC_ORIENTATION_MATRIX_HPP

#include "orientation_types.hpp"

#include <vector>

int pdag_index(int p, int row, int col);
int pdag_get(const std::vector<int>& pdag, int p, int row, int col);
void pdag_set(std::vector<int>* pdag, int p, int row, int col, int value);

bool has_any_edge(const std::vector<int>& pdag, int p, int a, int b);
bool has_undirected_edge(const std::vector<int>& pdag, int p, int a, int b);
bool has_directed_edge(const std::vector<int>& pdag, int p, int a, int b);
bool has_conflict_edge(const std::vector<int>& pdag, int p, int a, int b);

void set_no_edge(std::vector<int>* pdag, int p, int a, int b);
void set_undirected_edge(std::vector<int>* pdag, int p, int a, int b);
void set_directed_edge(std::vector<int>* pdag, int p, int from, int to);
void set_conflict_edge(std::vector<int>* pdag, int p, int a, int b);

std::vector<int> pdag_from_skeleton_adjacency(const std::vector<int>& adjacency, int p);

#endif
