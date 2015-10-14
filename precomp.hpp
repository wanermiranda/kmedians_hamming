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

#ifndef __OPENCV_PRECOMP_H__
#define __OPENCV_PRECOMP_H__
#include <algorithm>
#include <vector>
#include <opencv2/core/core.hpp>
#include <opencv2/core/mat.hpp>
#include <opencv2/core/types_c.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/highgui/highgui_c.h>
#include <opencv2/nonfree/nonfree.hpp>
#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/nonfree/nonfree.hpp>

namespace cv {
	namespace features2d {
		class LATCH {
		public:
			static Ptr<LATCH> create(int bytes = 32, bool rotationInvariance = true,
									 int half_ssd_size = 3);
		};

		class LATCHDescriptorExtractorImpl : public LATCH
		{
		public:
			enum { PATCH_SIZE = 32 };

			LATCHDescriptorExtractorImpl(int bytes = 32, bool rotationInvariance = true, int half_ssd_size = 3);

			void read( const FileNode& );
			void write( FileStorage& ) const;

			int descriptorSize() const;
			int descriptorType() const;
			int defaultNorm() const;

			void compute(Mat image, std::vector<KeyPoint>& keypoints, Mat &descriptors);

		protected:
			typedef void(*PixelTestFn)(const Mat input_image, const std::vector<KeyPoint>& keypoints, Mat& , const std::vector<int> &points, bool rotationInvariance, int half_ssd_size);
			void setSamplingPoints();
			int bytes_;
			PixelTestFn test_fn_;
			bool rotationInvariance_;
			int half_ssd_size_;


			std::vector<int> sampling_points_ ;
		};
	}
}
#endif
