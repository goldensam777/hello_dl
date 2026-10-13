#ifndef TOPO_H
#define TOPO_H

#include "../micrograd.h"
#include <stddef.h>

typedef struct {
    Value **nodes;
    size_t size;
    size_t capacity;
} TopologicalOrder;

int build_topological_order(Value *root, TopologicalOrder *order);
void free_topological_order(TopologicalOrder *order);

#endif