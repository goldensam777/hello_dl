#include "topo.h"
#include <stdlib.h>

static int contains_node(Value **nodes, size_t size, Value *target)
{
    for (size_t i = 0; i < size; i++) {
        if (nodes[i] == target) {
            return 1;
        }
    }

    return 0;
}

static int append_node(TopologicalOrder *order, Value *node)
{
    if (order->size == order->capacity) {
        size_t new_capacity = order->capacity == 0 ? 8 : order->capacity * 2;
        Value **new_nodes = realloc(order->nodes,
                                    new_capacity * sizeof(*new_nodes));
        if (new_nodes == NULL) {
            return 0;
        }

        order->nodes = new_nodes;
        order->capacity = new_capacity;
    }

    order->nodes[order->size++] = node;
    return 1;
}

static int visit_node(Value *node, Value ***visited, size_t *visited_size,
                      TopologicalOrder *order)
{
    if (node == NULL || contains_node(*visited, *visited_size, node)) {
        return 1;
    }

    Value **new_visited = realloc(*visited,
                                  (*visited_size + 1) * sizeof(**visited));
    if (new_visited == NULL) {
        return 0;
    }

    *visited = new_visited;
    (*visited)[(*visited_size)++] = node;

    for (size_t i = 0; i < VALUE_CHILDREN_COUNT; i++) {
        if (!visit_node(node->children[i], visited, visited_size, order)) {
            return 0;
        }
    }

    return append_node(order, node);
}

int build_topological_order(Value *root, TopologicalOrder *order)
{
    Value **visited = NULL;
    size_t visited_size = 0;
    int success;

    if (order == NULL) {
        return 0;
    }

    order->nodes = NULL;
    order->size = 0;
    order->capacity = 0;
    success = visit_node(root, &visited, &visited_size, order);
    free(visited);

    if (!success) {
        free_topological_order(order);
    }

    return success;
}

void free_topological_order(TopologicalOrder *order)
{
    if (order == NULL) {
        return;
    }

    free(order->nodes);
    order->nodes = NULL;
    order->size = 0;
    order->capacity = 0;
}
