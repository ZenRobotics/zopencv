%extend cv::flann::Index
{
    void knnSearch(CvMat* queries, CvMat* indices, CvMat* dists,
                   int knn, const SearchParams& params)
    {
        cv::Mat q(queries); // Thx C++
        cv::Mat i(indices);
        cv::Mat d(dists);
        $self->knnSearch(q, i, d, knn, params);
    }
};

%extend CvEM
{
    const CvMat* get_cov(int i)
    {
        return $self->get_covs()[i];
    }
}

//#define Mat CvMat
%ignore cv::flann::IndexFactory::createIndex;
%ignore cv::flann::Index::Index;
%ignore cv::flann::Index::knnSearch;
%ignore cv::flann::Index::radiusSearch;
%ignore hierarchicalClustering;
//%include "opencv2/flann/kdtree_index.h"
%include "opencv2/flann/defines.h"
%include "opencv2/flann/miniflann.hpp"
// %include "opencv2/flann/flann.hpp"
 //#undef Mat
