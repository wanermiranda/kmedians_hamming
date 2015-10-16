/*
 * Kmedians.h
 *
 *  Created on: Sep 30, 2015
 *      Author: waner
 */
#include <opencv2/core/core.hpp>
#include <opencv2/core/mat.hpp>
#include <opencv2/core/types_c.h>
#include <climits>
#include <stdlib.h>

#ifndef KMEDIANS_H_
#define KMEDIANS_H_


namespace cv {
    double kmedians(cv::InputArray _data, int K, cv::InputOutputArray _bestLabels,
                    cv::TermCriteria criteria, int attempts, int flags, cv::OutputArray _centers);

}


#endif /* KMEDIANS_H_ */