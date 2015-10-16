//
// Created by gorigan on 10/5/15.
//

#ifndef KMEDIANS_KMEDIANS_H
#define KMEDIANS_KMEDIANS_H


#include <stddef.h>
#include <iostream>
#include <random>
#include <opencv2/core/core.hpp>
#include <opencv2/core/mat.hpp>
using namespace cv;

class KmediansBinary {

    // [_points_size, _descriptor_size]
    Mat _points;
    // [_centers_size, _descriptor_size]
    Mat _centers;
    // [_centers_size]
    double *_mean_distance;
    // [_centers_size]
    int *_population_for_cluster;
    // [_points_size]
    int *_points_assigned_to_centers;
    // [_points_size, _centers_size]
    Mat _distance_matrix;
    // [_centers_size]
    bool *_has_center_moved;

    size_t _centers_size;

    int _max_tries = 10000;

    std::random_device _random_device;
    std::uniform_int_distribution<int> _int_distribution_generator;
public:

    KmediansBinary(Mat& points_, size_t centers_size, int max_tries, Mat& output_centers);
    ~KmediansBinary();

    void initCenters();

    void computeCentersMedian();

    void computeDistance();

    int makeAssignment();
};


#endif //KMEDIANS_KMEDIANS_H
