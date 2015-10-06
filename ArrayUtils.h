//
// Created by gorigan on 10/5/15.
//

#ifndef KMEDIANS_ARRAYUTILS_H
#define KMEDIANS_ARRAYUTILS_H

#include <stdlib.h>

template <typename T> T **alloc_2D_array(size_t M, size_t N) {
    T **array = (T **)malloc(sizeof (T) * M);
    if (array)
    {

        for (size_t i = 0; i < M; i++)
        {
            array[i] = (T *)malloc(sizeof(*array[i]) * N);
        }
    }
    return array;
}
template <typename T> T *alloc_1D_array(size_t N) {
    T *array = (T*)malloc(sizeof (T) * N);
    return array;
}
//template <typename T> T *()
#endif //KMEDIANS_ARRAYUTILS_H
