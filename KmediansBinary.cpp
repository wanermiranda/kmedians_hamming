#include <vector>
#include <set>
#include <cmath>
#include "KmediansBinary.h"
#include "ArrayUtils.h"


class kmediansDistanceSumComputer: public ParallelLoopBody {
public:
    Mat& _points;
    int* _distance_sum;
    kmediansDistanceSumComputer(
            Mat points,
            int *distance_sum) :
            _points(points),
            _distance_sum(distance_sum)
    {
    }

    void operator()(const cv::Range& range) const {
        const int begin = range.start;
        const int end = range.end;
        for (int n = begin; n < end; n++) {
            const int i = n;
            const uchar *point = _points.ptr<uchar>(i);
            for (int sample_idx = 0; sample_idx < _points.rows; sample_idx++) {

                const uchar *sample = _points.ptr<uchar>(sample_idx);

                _distance_sum[i] += normHamming(point, sample, _points.cols);
            }
        }

    }

private:
    kmediansDistanceSumComputer& operator=(const kmediansDistanceSumComputer&); // to quiet MSVC


};


class kmediansPPDistanceComputer: public ParallelLoopBody {
public:
    int _centers_size;
    Mat& _centers;
    Mat& _points;
    Mat& _distance_matrix;
    int *_population_for_cluster;
    bool *_has_center_moved;
    kmediansPPDistanceComputer(
            int centers_size,
            Mat centers,
            Mat points,
            Mat& distance_matrix,
            int *population_for_cluster,
            bool *has_center_moved) :

            _centers_size(centers_size),
            _centers(centers),
            _points(points),
            _distance_matrix(distance_matrix),
            _population_for_cluster(population_for_cluster),
            _has_center_moved(has_center_moved)
    {
    }

    void operator()(const cv::Range& range) const {
        const int begin = range.start;
        const int end = range.end;
        for (int n = begin; n < end; n++) {
            const int i = n;

            for (int c = 0; c < _centers_size; c++) {
                //if center didn't move, continue
                if (!_has_center_moved[c])
                    continue;

                //if hasn't _points in it, continue;
                if (_population_for_cluster[c] == 0) {
                    _distance_matrix.ptr<int>(i)[c] = INT_MAX;
                    continue;
                }

                int distance = 0;
                const uchar *point = _points.ptr<uchar>(i);
                const uchar *center = _centers.ptr<uchar>(c);

                distance = normHamming(point, center, _points.cols);

                if (std::isnan(distance))
                    distance = INT_MAX;

                _distance_matrix.ptr<uchar>(i)[c] = distance;
            }
        }

    }

private:
    kmediansPPDistanceComputer& operator=(const kmediansPPDistanceComputer&); // to quiet MSVC


};

KmediansBinary::KmediansBinary(Mat& points_, size_t centers_size, int max_tries, Mat& output_centers) {
    _points = points_;
    _centers_size = centers_size;
    _max_tries = max_tries;

    initCenters();
    std::cout << "Starting clustering." << std::endl;
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

        std::cout   << "iteration " << iteration << ", " << assignments_made
                    << " _points moved, " << centers_moved
                    << " _centers moved." << std::endl;
        iteration++;
    } while (assignments_made != 0 && iteration < _max_tries);
    std::cout << "Clustering done, cleaning..." << std::endl;

//    store variances also
    std::fill_n(_mean_distance, _centers_size, 0);
    for (int c = 0; c < _centers_size; c++) {
        int nbp = 0;
        for (int i = 0; i < _points.rows; i++) {
            if (_points_assigned_to_centers[i] == c) {
                _mean_distance[c] += _distance_matrix.at<double>(i,c);
                nbp++;
            }
        }
        if (nbp > 0)
            _mean_distance[c] /= (double) nbp;
    }

//    cleaning empty clusters
    Mat listOfCenters = cv::Mat(_centers_size, _points.cols, CV_8U);
    double listOfMeanDist[_centers_size];
    double listOfPopulation[_centers_size];
    int listSize = 0;
    // bringing the non populated _centers up to the front
    for (int c = 0; c < _centers_size; c++) {
        if (_population_for_cluster[c] > 0) {
            listSize++;
            _centers.row(c).copyTo(listOfCenters.row(listSize));
            listOfMeanDist[listSize] = _mean_distance[c];
            listOfPopulation[listSize] = _population_for_cluster[c];
        }
    }
    _centers.release();

    _centers = cv::Mat(_centers_size, _points.cols, CV_8U);
    std::fill_n(_mean_distance, _centers_size, 0);
    std::fill_n(_population_for_cluster, _centers_size, 0);
    for (int c = 0; c < listSize; c++) {
        listOfCenters.row(c).copyTo(_centers.row(c));
        _mean_distance[c] = listOfMeanDist[c];
        _population_for_cluster[c] = listOfPopulation[c];
    }

    listOfCenters.release();

//  Assign the min point representing the _centers
    std::set<int> listP;
    for (int c = 0; c < _centers_size; c++) {
        double min = INFINITY;
        int pointMin = -1;
//           getting the minimum point of the center
        for (int i = 0; i < _points.rows; i++) {
            if (_points_assigned_to_centers[i] == c) {
                if (min > _distance_matrix.ptr<int>(i)[c]) {
                    min = _distance_matrix.ptr<int>(i)[c];
                    pointMin = i;
                }
            }
        }
        if (listP.find(pointMin) == listP.end())
            listP.insert(pointMin);

        if (pointMin != -1) {
                _points.row(pointMin).copyTo(_centers.row(c));
        }

    }

    std::cout << "Cleaning done. Clusters are ready." << std::endl;

    _centers.copyTo(output_centers);
}

void KmediansBinary::computeCentersMedian() {
    for (int c = 0; c < _centers_size; c++) {

        if (!_has_center_moved[c])
            continue;
        if (_population_for_cluster[c] == 0) {
            _centers.row(c) = Scalar(-1);
            continue;
        }


        // Computes the hamming distances from all to all _points
        int distanceSums[_points.rows];

        parallel_for_(Range(0, _points.rows), kmediansDistanceSumComputer(_points,distanceSums));

        int newCenter = -1;
        int shortDist = std::numeric_limits<int>::max();
        // Capture the median point,
        // The point that minimizes the sum of all distances in the same cluster
        for (int i = 0; i < _points.rows; i++) {
            if (distanceSums[i] < shortDist) {
                shortDist = distanceSums[i];
                newCenter = i;
            }
        }
        // Copying the point to the center position
        uchar *newCenterDesc = _points.ptr<uchar>(newCenter);
        uchar *center = _centers.ptr<uchar>(c);
        center = newCenterDesc;

    }
}

void KmediansBinary::initCenters() {
    _centers = cv::Mat(_centers_size, _points.cols, CV_8U);
    _mean_distance = alloc_1D_array<double>(_centers_size);
    _population_for_cluster =  alloc_1D_array<int>(_centers_size);;

    //random assignment
    std::cout << "Initializing _centers..." << std::endl;
    _points_assigned_to_centers = alloc_1D_array<int>(_points.rows);
    std::fill_n(_points_assigned_to_centers, _points.rows, -1);

    //-1 == no assignment
    std::set<int> generated_numbers;
    _int_distribution_generator = std::uniform_int_distribution<int>(_points.rows);

    //pick a random point for each cluster
    for (int i = 0; i < _centers_size; i++) {
        int indexPoint = _int_distribution_generator(_random_device) % _points.rows;
        while (generated_numbers.find(indexPoint) != generated_numbers.end())
            indexPoint = _int_distribution_generator(_random_device) % _points.rows;
        generated_numbers.insert(indexPoint);
        _points_assigned_to_centers[indexPoint] = i;

        _population_for_cluster[i]++;
        if (i % (_centers_size / 20 + 1) == 0)
            std::cout << "." << std::endl;
    }

    std::cout << std::endl << std::endl;

    //distance matrix and has moved
    _distance_matrix = Mat(_points.rows, _centers_size, CV_32S);
    _has_center_moved = alloc_1D_array<bool>(_centers_size);
    std::fill_n(_has_center_moved, _centers_size, true);
    std::cout << "Centers randomly initialized." << std::endl;
}

void KmediansBinary::computeDistance() {

    parallel_for_(Range(0, _points.rows), kmediansPPDistanceComputer(_centers_size, _centers, _points, _distance_matrix,
                                                                     _population_for_cluster, _has_center_moved));
}

int KmediansBinary::makeAssignment() {

    int nbm = 0;

    std::fill_n(_population_for_cluster, _centers_size, 0);
    std::fill_n(_has_center_moved, _centers_size, false);


    for (int i = 0; i < _points.rows; i++) {
        //find the minimal distance
        int indexMin = 0;
        double dist = _distance_matrix.ptr<int>(i)[0];
        for (int m = 0; m < _centers_size; m++) {
            if (_distance_matrix.ptr<int>(i)[m] < dist) {
                dist = _distance_matrix.ptr<int>(i)[m];
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

KmediansBinary::~KmediansBinary() {
    _points.release();
    // [_centers_size, _descriptor_size]
    _centers.release();
    // [_centers_size]
    delete _mean_distance;
    // [_centers_size]
    delete _population_for_cluster;
    // [_points_size]
    delete _points_assigned_to_centers;
    // [_points_size, _centers_size]
    _distance_matrix.release();
    // [_centers_size]
    delete _has_center_moved;
}
