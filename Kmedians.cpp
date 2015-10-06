#include <vector>
#include <set>
#include <cmath>
#include "Kmedians.h"
#include "ArrayUtils.h"

Kmedians::Kmedians(int **points_, size_t centers_size, size_t points_size, size_t descriptor_size, int max_tries) {
    _points_size = points_size;
    _descriptor_size = descriptor_size;
    _points = points_;
    _centers_size = centers_size;
    _max_tries = max_tries;

    initCenters();
    std::cout << "Starting clustering.";
    int assignments_made = 0;
    int iteration = 1;
    do {
        computeCentersMedian();

        computeDistance();

        assignments_made = makeAssignment();

        int centers_moved = 0;
        for (int c = 0; c < _centers_size; c++)
            if (_has_center_moved[c])
                centers_moved++;

        std::cout   << "iteration " << iteration << ", " << assignments_made + " _points moved, " << centers_moved
                    << " _centers moved.";
        iteration++;
    } while (assignments_made != 0 && iteration < _max_tries);
    std::cout << "Clustering done, cleaning...";

//    store variances also
    std::fill_n(_mean_distance, _centers_size, 0);
    for (int c = 0; c < _centers_size; c++) {
        int nbp = 0;
        for (int i = 0; i < _points_size; i++) {
            if (_points_assigned_to_centers[i] == c) {
                _mean_distance[c] += _distance_matrix[i][c];
                nbp++;
            }
        }
        if (nbp > 0)
            _mean_distance[c] /= (double) nbp;
    }

//    cleaning empty clusters
    double **listOfCenters = alloc_2D_array<double>(_centers_size, _descriptor_size);
    double listOfMeanDist[_centers_size];
    double listOfPopulation[_centers_size];
    int listSize = 0;
    // bringing the non populated _centers up to the front
    for (int c = 0; c < _centers_size; c++) {
        if (_population_for_cluster[c] > 0) {
            listSize++;
            listOfCenters[listSize] = _centers[c];
            listOfMeanDist[listSize] = _mean_distance[c];
            listOfPopulation[listSize] = _population_for_cluster[c];
        }
    }
    delete _centers;
    _centers = alloc_2D_array<double>(_centers_size, _descriptor_size);
    std::fill_n(_mean_distance, _centers_size, 0);
    std::fill_n(_population_for_cluster, _centers_size, 0);
    for (int c = 0; c < listSize; c++) {
        _centers[c] = listOfCenters[c];
        _mean_distance[c] = listOfMeanDist[c];
        _population_for_cluster[c] = listOfPopulation[c];
    }

    delete listOfCenters;

//  Assign the min point representing the _centers
    std::set<int> listP;
    for (int c = 0; c < _centers_size; c++) {
        double min = INFINITY;
        int pointMin = -1;
//           getting the minimum point of the center
        for (int i = 0; i < _points_size; i++) {
            if (_points_assigned_to_centers[i] == c) {
                if (min > _distance_matrix[i][c]) {
                    min = _distance_matrix[i][c];
                    pointMin = i;
                }
            }
        }
        if (listP.find(pointMin) == listP.end())
            listP.insert(pointMin);
        if (pointMin != -1) {
            for (int j = 0; j < _descriptor_size; j++)
                _centers[c][j] = _points[pointMin][j];
        }
        else {
            for (int j = 0; j < _descriptor_size; j++)
                _centers[c][j] = _points[pointMin][j];
        }
    }

    std::cout << "Cleaning done. Clusters are ready.";
}

void Kmedians::computeCentersMedian() {
    for (int c = 0; c < _centers_size; c++) {

        if (!_has_center_moved[c])
            continue;
        if (_population_for_cluster[c] == 0) {
            std::fill_n(_centers[c], _points_size, NAN);
            continue;
        }


        // Computes the hamming distances from all to all _points
        int distanceSums[_points_size];

        for (int i = 0; i < _points_size; i++) {
            for (int j = i + 1; j < _points_size; j++) {
                int distance = 0;
                for (int d = 0; d < _descriptor_size; d++) {
                    int *descI = _points[i];
                    int *descJ = _points[j];
                    distance += hammingDistance(((descI[d]) & 0xff), ((descJ[d]) & 0xff));
                }
                distanceSums[j] += distance;
            }
        }

        int newCenter = -1;
        int shortDist = std::numeric_limits<int>::max();
        // Capture the median point,
        // The point that minimizes the sum of all distances in the same cluster
        for (int i = 0; i < _points_size; i++) {
            if (distanceSums[i] < shortDist) {
                shortDist = distanceSums[i];
                newCenter = i;
            }
        }
        // Copying the point to the center position
        int *newCenterDesc = _points[newCenter];
        for (int j = 0; j < _descriptor_size; j++) {
            _centers[c][j] = newCenterDesc[j];
        }

    }
}

void Kmedians::initCenters() {
    _centers = alloc_2D_array<double>(_centers_size, _descriptor_size);
    _mean_distance = alloc_1D_array<double>(_centers_size);
    _population_for_cluster =  alloc_1D_array<int>(_centers_size);;

    //random assignment
    std::cout << "Initializing _centers...";
    _points_assigned_to_centers = alloc_1D_array<int>(_points_size);
    std::fill_n(_points_assigned_to_centers, _points_size, -1);

    //-1 == no assignment
    std::set<int> generated_numbers;
    _int_distribution_generator = std::uniform_int_distribution<int>(_points_size);

    //pick a random point for each cluster
    for (int i = 0; i < _centers_size; i++) {
        int indexPoint = _int_distribution_generator(_random_device);
        while (generated_numbers.find(indexPoint) != generated_numbers.end())
            indexPoint = _int_distribution_generator(_random_device);
        generated_numbers.insert(indexPoint);
        _points_assigned_to_centers[indexPoint] = i;

        _population_for_cluster[i]++;
        if (i % (_centers_size / 20 + 1) == 0)
            std::cout << ".";
    }

    std::cout << std::endl;

    //distance matrix and has moved
    _distance_matrix = alloc_2D_array<double>(_points_size, _centers_size);
    _has_center_moved = alloc_1D_array<bool>(_centers_size);
    std::fill_n(_has_center_moved, _centers_size, true);
    std::cout << "Centers randomly initialized.";
}


void Kmedians::computeDistance() {

    for (int n = 0; n < _points_size; n++) {
        const int i = n;

        for (int c = 0; c < _centers_size; c++) {
            //if center didn't move, continue
            if (!_has_center_moved[c])
                continue;

            //if hasn't _points in it, continue;
            if (_population_for_cluster[c] == 0) {
                _distance_matrix[i][c] = INFINITY;
                continue;
            }

            double distance = 0;
            int *p = _points[i];
            double *center = _centers[c];

            for (int d = 0; d < _descriptor_size; d++) {

                //Verifying if the center is setted to NAN to avoid further arithmetical errors
                if (std::isnan(center[d])) {
                    distance = NAN;
                    break;
                }
                else {
                    distance += hammingDistance((p[d] & 0xff), (((int) center[d]) & 0xff));
                }
            }

            if (std::isnan(distance))
                distance = INFINITY;
            _distance_matrix[i][c] = distance;
        }
    }
}

int Kmedians::makeAssignment() {

    int nbm = 0;

    std::fill_n(_population_for_cluster, _centers_size, 0);
    std::fill_n(_has_center_moved, _centers_size, false);


    for (int i = 0; i < _points_size; i++) {
        //find the minimal distance
        int indexMin = 0;
        double dist = _distance_matrix[i][0];
        for (int m = 0; m < _centers_size; m++) {
            if (_distance_matrix[i][m] < dist) {
                dist = _distance_matrix[i][m];
                indexMin = m;
            }
        }

        _population_for_cluster[indexMin]++;

        //compare to the original
        int oldIndex = _points_assigned_to_centers[i];
        if (oldIndex != indexMin) {
            //got one more move
            nbm++;
            //_centers will change
            _has_center_moved[indexMin] = true;
            if (oldIndex != -1)
                _has_center_moved[oldIndex] = true;

            //make assignment
            _points_assigned_to_centers[i] = indexMin;

        }
    }
    return nbm;
}

int Kmedians::hammingDistance(int x, int y) {
    int dist = 0;
    int val = x ^y; // XOR

    // Count the number of set bits
    while (val != 0) {
        ++dist;
        val &= val - 1;
    }

    return dist;
}


Kmedians::~Kmedians() {
    delete _points;
    // [_centers_size, _descriptor_size]
    delete _centers;
    // [_centers_size]
    delete _mean_distance;
    // [_centers_size]
    delete _population_for_cluster;
    // [_points_size]
    delete _points_assigned_to_centers;
    // [_points_size, _centers_size]
    delete _distance_matrix;
    // [_centers_size]
    delete _has_center_moved;
}
