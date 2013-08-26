// -*- mode: c++ -*-

%{

#include <limits>

void cvZPredictRTrees(CvRTrees* rtrees,
        CvArr* features,
        CvArr* responses
        )
{
    CvMat m;
    CvMat* mp;
    mp = cvGetMat(features, &m);
    CvSize sz = cvGetSize(mp);
    for (int r = 0; r < sz.height; r++)
    {
        CvMat sub;
        CvMat* subp;
        subp = cvGetSubRect(mp, &sub, cvRect(0, r, sz.width, 1));
        double res = rtrees->predict_prob(subp, 0);
        cvSet2D(responses, r, 0, cvScalar(res));
    }
}

void cvZPredictRTreesMulti(CvRTrees* rtrees,
                           CvMat* features,
                           CvMat* responses_)
{
    using namespace cv;
    CvSize sz = cvGetSize(features);
    CvSize rsz = cvGetSize(responses_);
    int ntrees = rtrees->get_tree_count();
    cvSet(responses_, cvScalar(0));
    Mat1f responses(responses_);
    for (int row = 0; row < sz.height; row++)
    {
        CvMat sub;
        CvMat* subp;
        subp = cvGetRow(features, &sub, row);

        for (int tree = 0; tree < ntrees;tree++)
        {
            CvDTreeNode* predicted = rtrees->get_tree(tree)->predict(subp);
            int cls = predicted->class_idx;
            CV_Assert(cls >= 0);
            CV_Assert(cls < rsz.width);
            responses.at<float>(row, cls) += 1;
        }
    }
}

void cvZHardMax (CvMat* _A)
{
    using namespace cv;
    Mat1f A (_A);
    for (int i = 0; i < A.rows; ++i)
    {
        int max_j = 0;
        for (int j = 1; j < A.cols; ++j)
            if (A(i,j) > A(i, max_j))
                max_j = j;
        A.row(i) = 0.0f;
        A(i, max_j) = 1.0f;
    }
}

void cvZNormalizeProbability (CvMat* _A)
{
    using namespace cv;
    Mat1f A (_A);
    for (int i = 0; i < A.rows; ++i)
    {
        double f = 1.0 / sum(A.row(i))[0];
        A.row(i) *= f;
    }
}

void cvZSoftMax (CvMat* _A, float alpha)
{
    using namespace cv;
    Mat1f A (_A);
    A *= alpha;
    exp(A, A);
    cvZNormalizeProbability(_A);
}

void cvZThresholdMax (CvMat* _A, CvMat* _th)
{
    using namespace cv;
    Mat1f A (_A);
    Mat1f th (_th);

    for (int i = 0; i < A.rows; ++i)
    {
        int max_j = -1;
        float max_p = -std::numeric_limits<float>::infinity();
        for (int j = 0; j < A.cols; ++j)
            if (A(i,j) >= th(j) && A(i,j) > max_p)
            {
                max_p = A(i,j);
                max_j = j;
            }
        CV_Assert(max_j >= 0);
        A.row(i) = 0.0f;
        A(i, max_j) = 1.0f;
    }
}

/** Given arrays
 * src(nrows, ncols), dst(nselected, ncols) and rows(nrows,1)
 * where nselected == cvCountNonZero(rows),
 * take all rows i of src for which rows(i,0) is nonzero
 * and copy them to dst in order.
 *
 * Useful for selecting a subset of data for OpenCV's machine
 * learning functions.
 */
void cvZSelectRows(CvArr* src_, CvArr* dst_, CvArr* rows_)
{
    CvMat msrc, mdst, mrows;
    CvMat* src = cvGetMat(src_, &msrc);
    CvMat* dst = cvGetMat(dst_, &mdst);
    CvMat* rows = cvGetMat(rows_, &mrows);
    CvSize sz = cvGetSize(src);
    CvSize dsz = cvGetSize(dst);
    CvSize rsz = cvGetSize(rows);

    CvMat msrect;
    CvMat mdrect;

    CV_Assert(sz.width == dsz.width);
    CV_Assert(sz.height == rsz.height);
    int index = 0;
    for (int r = 0; r < sz.height; r++)
    {
        // Insanely ineffective but ok for us right now
        if (cvGet2D(rows_, r, 0).val[0])
        {
            CV_Assert(index < dsz.height);
            CvMat* srect = cvGetSubRect(src, &msrect,
                                cvRect(0, r, sz.width, 1));
            CvMat* drect = cvGetSubRect(dst, &mdrect,
                                cvRect(0, index, sz.width, 1));

            cvCopy(srect, drect);

            index++;
        }
    }
}

%}

void cvZPredictRTrees(CvRTrees* rtrees,
                      CvMat* features,
                      CvMat* responses);

// Use rtrees to classify the input data features, returning
// votes per class in responses_. Each row of features is
// one input data point and columns must match what the trees
// expect, the responses_ must have as many rows as the input
// and as many columns as there are classes in the trees. On
// return responses_ will have votes per class in the columns.
//
// This is our own code since the built-in OpenCV prediction
// interface doesn't return the vote counts if there are more
// than two classes.
void cvZPredictRTreesMulti(CvRTrees* rtrees,
                           CvMat* features,
                           CvMat* responses_);

// Replace precisely one maximum on each row of A with 1.0 and set the rest
// of the row to 0.0. Used to transform class probabilities from a classifier
// (one row of A contains class probabilities for an object) to a sharp
// prediction that predicts a most likely class for each object with
// probability 1.0 (when there are equiprobable classes, picks the
// first of the likeliest classes).
// XXX(tt): Only works with floats now.
void cvZHardMax (CvMat* A);

// Applies the softmax function a_ij -> exp(alpha*a_ij)/sum_j(exp(alpha*a_ij)).
// to rows of A. Can be used e.g. to approximate class probabilities from
// relative proportions of votes in random trees provided that alpha is
// fitted with training data.
// XXX(tt): Only works with floats now.
void cvZSoftMax (CvMat* A, float alpha);

// cvZThresholdMax converts a discrete probability distribution to
// a single class prediction with required confidence. The rows of A are
// considered to contain the discrete class probabilities from a classifier
// to be converted to predictions and th contains the classwise thresholds.

// cvZThresholdMax chooses the most likely class with probability above
// classwise threshold. The minimum threshold must be <= 0.0 so that
// there is always at least one admissible class.
//
// XXX(tt): Only works with floats now.
void cvZThresholdMax (CvMat* _A, CvMat* _th);

// Divides each row of A by its sum to get a probability distribution
// (assuming A >= 0).
// XXX(tt): Only works with floats now.
void cvZNormalizeProbability (CvMat* A);

void cvZSelectRows(CvMat* src_, CvMat* dst_, CvMat* rows_);

// Make a RTParams structure for training a random tree. The default wrapper
// does not support passing a prior weight array.
%newobject cvZRTParams;
CvRTParams* cvZRTParams (int max_depth,
                         int min_sample_count,
                         float regression_accuracy,
                         bool use_surrogates,
                         int max_categories,
                         JavaFloatArray priors,
                         bool calc_var_importance,
                         int nactive_vars,
                         int max_num_of_trees_in_the_forest,
                         float forest_accuracy,
                         int termcrit_type);

%typemap(javadestruct, methodname="delete", methodmodifiers="public synchronized") CvRTParams %{
{
    if(swigCPtr != 0 && swigCMemOwn) {
        swigCMemOwn = false;
        zopencvJNI.cvReleaseRTParams(swigCPtr, this);
    }
    swigCPtr = 0;
}
%}

%{

CvRTParams* cvZRTParams (int max_depth,
                         int min_sample_count,
                         float regression_accuracy,
                         bool use_surrogates,
                         int max_categories,
                         JavaFloatArray priors,
                         bool calc_var_importance,
                         int nactive_vars,
                         int max_num_of_trees_in_the_forest,
                         float forest_accuracy,
                         int termcrit_type)
{
    float* p = 0;
    if (priors.size > 0)
    {
        p = new float [priors.size];
        for (unsigned i = 0; i < priors.size; ++i)
            p[i] = priors.ptr[i];
    }
    return new CvRTParams (max_depth,
                           min_sample_count,
                           regression_accuracy,
                           use_surrogates,
                           max_categories,
                           p,
                           calc_var_importance,
                           nactive_vars,
                           max_num_of_trees_in_the_forest,
                           forest_accuracy,
                           termcrit_type);
}

void cvReleaseRTParams (CvRTParams* c)
{
    if (!c) return;
    if (c->priors)
        delete [] c->priors;
    delete c;
}

%}

