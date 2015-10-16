/*M///////////////////////////////////////////////////////////////////////////////////////
 //
 //  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
 //
 //  By downloading, copying, installing or using the software you agree to this license.
 //  If you do not agree to this license, do not download, install,
 //  copy or use the software.
 //
 //
 //                          License Agreement
 //                For Open Source Computer Vision Library
 //
 // Copyright (C) 2000-2008, Intel Corporation, all rights reserved.
 // Copyright (C) 2009, Willow Garage Inc., all rights reserved.
 // Copyright (C) 2013, OpenCV Foundation, all rights reserved.
 // Third party copyrights are property of their respective owners.
 //
 // Redistribution and use in source and binary forms, with or without modification,
 // are permitted provided that the following conditions are met:
 //
 //   * Redistribution's of source code must retain the above copyright notice,
 //     this list of conditions and the following disclaimer.
 //
 //   * Redistribution's in binary form must reproduce the above copyright notice,
 //     this list of conditions and the following disclaimer in the documentation
 //     and/or other materials provided with the distribution.
 //
 //   * The name of the copyright holders may not be used to endorse or promote products
 //     derived from this software without specific prior written permission.
 //
 // This software is provided by the copyright holders and contributors "as is" and
 // any express or implied warranties, including, but not limited to, the implied
 // warranties of merchantability and fitness for a particular purpose are disclaimed.
 // In no event shall the Intel Corporation or contributors be liable for any direct,
 // indirect, incidental, special, exemplary, or consequential damages
 // (including, but not limited to, procurement of substitute goods or services;
 // loss of use, data, or profits; or business interruption) however caused
 // and on any theory of liability, whether in contract, strict liability,
 // or tort (including negligence or otherwise) arising in any way out of
 // the use of this software, even if advised of the possibility of such damage.
 //
 //M*/

#include "precomp.hpp"
#include <opencv2/core/core.hpp>
#include <random>
#include <set>
#include <iostream>
#include "Kmedians__.h"
#include <climits>

////////////////////////////////////////// kmedians ////////////////////////////////////////////

namespace cv {


    static void pickUpRandomCenters(const Mat& _data, Mat& _out_centers, int K,
                                    RNG& rng) {
        std::set<int> generated_numbers;
        std::uniform_int_distribution<int> _int_distribution_generator = std::uniform_int_distribution<int>(_data.rows);
        std::random_device _random_device;
        //pick a random point for each cluster
        for (int i = 0; i < K; i++) {
            int indexPoint = (_int_distribution_generator(_random_device) % _data.rows);
            while (generated_numbers.find(indexPoint) != generated_numbers.end())
                indexPoint = (_int_distribution_generator(_random_device) % _data.rows);
            generated_numbers.insert(indexPoint);
            std::cout <<  "i:" << i << " index:" << indexPoint<< std::endl;
            _out_centers.row(i) = _data.row(indexPoint);

        }
    }

    class kmediansPPDistanceComputer: public ParallelLoopBody {
    public:
        kmediansPPDistanceComputer(int *_tdist2, const uchar *_data,
                                   const int *_dist, int _dims, size_t _step, size_t _stepci) :
                tdist2(_tdist2), data(_data), dist(_dist), dims(_dims), step(_step), stepci(
                _stepci) {
        }

        void operator()(const cv::Range& range) const {
            const int begin = range.start;
            const int end = range.end;

            for (int i = begin; i < end; i++) {
                tdist2[i] = std::min(normHamming(data + step * i, data + stepci, dims),
                                     dist[i]);
            }
        }

    private:
        kmediansPPDistanceComputer& operator=(const kmediansPPDistanceComputer&); // to quiet MSVC

        int *tdist2;
        const uchar *data;
        const int *dist;
        const int dims;
        const size_t step;
        const size_t stepci;
    };

/*
 k-means center initialization using the following algorithm:
 Arthur & Vassilvitskii (2007) k-means++: The Advantages of Careful Seeding
 */
    static void generateCentersPP(const Mat& _data, Mat& _out_centers, int K,
                                  RNG& rng, int trials) {
        int i, j, k, dims = _data.cols, N = _data.rows;
        const uchar* data = _data.ptr<uchar>(0);
        size_t step = _data.step / sizeof(data[0]);
        std::vector<int> _centers(K);
        int* centers = &_centers[0];
        std::vector<int> _dist(N * 3);
        int* dist = &_dist[0], *tdist = dist + N, *tdist2 = tdist + N;
        int sum0 = 0;

        centers[0] = (unsigned) rng % N;

        for (i = 0; i < N; i++) {
            dist[i] = normHamming(data + step * i, data + step * centers[0], dims);
            sum0 += dist[i];
        }

        for (k = 1; k < K; k++) {
            int bestSum = INT_MAX;
            int bestCenter = -1;

            for (j = 0; j < trials; j++) {
                int p = (int) rng * sum0, s = 0;
                for (i = 0; i < N - 1; i++)
                    if ((p -= dist[i]) <= 0)
                        break;
                int ci = i;

                parallel_for_(Range(0, N),
                              kmediansPPDistanceComputer(tdist2, data, dist, dims, step,
                                                         step * ci));
                for (i = 0; i < N; i++) {
                    s += tdist2[i];
                }

                if (s < bestSum) {
                    bestSum = s;
                    bestCenter = ci;
                    std::swap(tdist, tdist2);
                }
            }
            centers[k] = bestCenter;
            sum0 = bestSum;
            std::swap(dist, tdist);
        }

        for (k = 0; k < K; k++) {
            const uchar* src = data + step * centers[k];
            uchar* dst = _out_centers.ptr<uchar>(k);
            for (j = 0; j < dims; j++)
                dst[j] = src[j];
        }
    }

    class kmediansDistanceComputer: public ParallelLoopBody {
    public:
        kmediansDistanceComputer(int *_distances, int *_labels, const Mat& _data,
                                 const Mat& _centers) :
                distances(_distances), labels(_labels), data(_data), centers(
                _centers) {
        }

        void operator()(const Range& range) const {
            const int begin = range.start;
            const int end = range.end;
            const int K = centers.rows;
            const int dims = centers.cols;

            // Computing distance from all centers, retrieving the nearest center
            for (int i = begin; i < end; ++i) {
                const uchar *sample = data.ptr<uchar>(i);
                int k_best = 0;
                double min_dist = DBL_MAX;

                for (int k = 0; k < K; k++) {
                    const uchar* center = centers.ptr<uchar>(k);
                    const int dist = normHamming(sample, center, dims);

                    if (min_dist > dist) {
                        min_dist = dist;
                        k_best = k;
                    }
                }

                distances[i] = min_dist;
                labels[i] = k_best;
            }
        }

    private:
        kmediansDistanceComputer& operator=(const kmediansDistanceComputer&); // to quiet MSVC

        int *distances;
        int *labels;
        const Mat& data;
        const Mat& centers;
    };

    double kmedians(InputArray _data, int K, InputOutputArray _bestLabels,
                    TermCriteria criteria, int attempts, int flags, OutputArray _centers) {
        const int SPP_TRIALS = 3;
        Mat data0 = _data.getMat();
        bool isrow = data0.rows == 1 && data0.channels() > 1;
        int N = !isrow ? data0.rows : data0.cols;
        int dims = (!isrow ? data0.cols : 1) * data0.channels();
        int type = data0.depth();

        attempts = std::max(attempts, 1);
        CV_Assert(data0.dims <= 2 && type == CV_8U && K > 0);
        CV_Assert(N >= K);

        Mat data(N, dims, CV_8U, data0.ptr(),
                 isrow ? dims * sizeof(uchar) : static_cast<size_t>(data0.step));

        _bestLabels.create(N, 1, CV_32S, -1, true);

        Mat _labels, best_labels = _bestLabels.getMat();
        if (flags & CV_KMEANS_USE_INITIAL_LABELS) {
            CV_Assert(
                    (best_labels.cols == 1 || best_labels.rows == 1) && best_labels.cols*best_labels.rows == N && best_labels.type() == CV_32S && best_labels.isContinuous());
            best_labels.copyTo(_labels);
        } else {
            if (!((best_labels.cols == 1 || best_labels.rows == 1)
                  && best_labels.cols * best_labels.rows == N
                  && best_labels.type() == CV_32S && best_labels.isContinuous()))
                best_labels.create(N, 1, CV_32S);
            _labels.create(best_labels.size(), best_labels.type());
        }
        int* labels = _labels.ptr<int>();

        Mat centers(K, dims, type), old_centers(K, dims, type), temp(1, dims, type);
        std::vector<int> counters(K);
        std::vector<Vec2b> _box(dims);
        Vec2b* box = &_box[0];
        double best_compactness = DBL_MAX, compactness = 0;
        RNG& rng = theRNG();
        int a, iter, i, j, k;

        if (criteria.type & TermCriteria::EPS)
            criteria.epsilon = std::max(criteria.epsilon, 0.);
        else
            criteria.epsilon = FLT_EPSILON;
        criteria.epsilon *= criteria.epsilon;

        if (criteria.type & TermCriteria::COUNT)
            criteria.maxCount = std::min(std::max(criteria.maxCount, 2), 100);
        else
            criteria.maxCount = 100;

        if (K == 1) {
            attempts = 1;
            criteria.maxCount = 2;
        }

        const uchar* sample = data.ptr<uchar>(0);
        for (j = 0; j < dims; j++)
            box[j] = Vec2f(sample[j], sample[j]);

        for (i = 1; i < N; i++) {
            sample = data.ptr<uchar>(i);
            for (j = 0; j < dims; j++) {
                uchar v = sample[j];
                box[j][0] = std::min(box[j][0], v);
                box[j][1] = std::max(box[j][1], v);
            }
        }

        for (a = 0; a < attempts; a++) {
            int max_center_shift = INT_MAX;
            for (iter = 0;;) {
                swap(centers, old_centers);

                if (iter == 0 && (a > 0 || !(flags & KMEANS_USE_INITIAL_LABELS))) {
                    if (flags & KMEANS_PP_CENTERS)
                        generateCentersPP(data, centers, K, rng, SPP_TRIALS);
                    else {
                        pickUpRandomCenters(data, centers, K, rng);
                    }
                } else {
                    if (iter == 0 && a == 0
                        && (flags & KMEANS_USE_INITIAL_LABELS)) {
                        for (i = 0; i < N; i++)
                            CV_Assert((unsigned )labels[i] < (unsigned )K);
                    }

//                     compute centers
                    centers = Scalar(0);
                    for (k = 0; k < K; k++)
                        counters[k] = 0;


                }

                if (++iter == MAX(criteria.maxCount, 2)
                    || max_center_shift <= criteria.epsilon)
                    break;

                // assign labels
                Mat dists(1, N, CV_32S);
                int* dist = dists.ptr<int>(0);
                parallel_for_(Range(0, N),
                              kmediansDistanceComputer(dist, labels, data, centers));
                compactness = 0;
                for (i = 0; i < N; i++) {
                    compactness += dist[i];
                }
            }

            if (compactness < best_compactness) {
                best_compactness = compactness;
                if (_centers.needed())
                    centers.copyTo(_centers);
                _labels.copyTo(best_labels);
            }
        }

        return best_compactness;
    }
}