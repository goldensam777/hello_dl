#include "micrograd.h"
#include <stdio.h>
#include <stdlib.h>


int main(void)
{
    const size_t layer_sizes[] = {100, 32, 16, 1};
    Value *inputs[100] = {NULL};
    MLP mlp = {0};
    ValueArray outputs;

    srand(42);

    for (size_t i = 0; i < 100; i++) {
        inputs[i] = init_value((double)(i + 1) / 100.0, 0.0, '\0', "input",
                               NULL);
        if (inputs[i] == NULL) {
            for (size_t j = 0; j < i; j++) {
                free_value_graph(inputs[j]);
            }
            return EXIT_FAILURE;
        }
    }

    if (!mlp_init(&mlp, layer_sizes, 3)) {
        for (size_t i = 0; i < 100; i++) {
            free_value_graph(inputs[i]);
        }
        return EXIT_FAILURE;
    }

    outputs = mlp_forward(&mlp, inputs, 100);
    if (outputs.values == NULL || outputs.size != 1) {
        mlp_free(&mlp);
        for (size_t i = 0; i < 100; i++) {
            free_value_graph(inputs[i]);
        }
        return EXIT_FAILURE;
    }

    value_backward(outputs.values[0]);

    printf("MLP architecture: 100 -> 32 -> 16 -> 1\n");
    printf("Parameters: %zu\n", mlp.params.size);
    printf("Output: %g\n", outputs.values[0]->data);
    printf("First weight gradient: %g\n", mlp.params.values[0]->grad);

    free_value_graphs(outputs.values, outputs.size);
    value_array_free(&outputs);
    mlp_free(&mlp);

    return 0;
}