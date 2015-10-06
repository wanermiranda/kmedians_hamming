//
// Created by gorigan on 10/5/15.
//

#ifndef KMEDIANS_KMEDIANS_H
#define KMEDIANS_KMEDIANS_H


#include <stddef.h>
#include <iostream>
#include <random>

class Kmedians {

    // [_points_size, _descriptor_size]
    int **_points;
    // [_centers_size, _descriptor_size]
    double **_centers;
    // [_centers_size]
    double *_mean_distance;
    // [_centers_size]
    int *_population_for_cluster;
    // [_points_size]
    int *_points_assigned_to_centers;
    // [_points_size, _centers_size]
    double **_distance_matrix;
    // [_centers_size]
    bool *_has_center_moved;

    size_t _descriptor_size;
    size_t _centers_size;
    size_t _points_size;

    int _max_tries = 10000;

    std::random_device _random_device;
    std::uniform_int_distribution<int> _int_distribution_generator;
public:

    Kmedians(int **points_, size_t centers_size, size_t points_size, size_t descriptor_size, int max_tries);
    ~Kmedians();

    void initCenters();

    void computeCentersMedian();

    int hammingDistance(int x, int y);

    void computeDistance();

    int makeAssignment();
};


#endif //KMEDIANS_KMEDIANS_H
