// Miscellaneous functions
void cvZMahalanobis2(CvMat* x_, CvMat* mu_, CvMat* icovar_, CvMat* dst_);
void cvZConvexHull2(CvMat* points, CvMat* hull, int clockwise);
jobjectArray cvExtractSURFSimple(CvMat* img, CvMat* mask,
                    CvSURFParams* params);
jintArray cvHaarDetectObjectSimple(CvMat* img,
                CvHaarClassifierCascade* cascade,
                double scale_factor,
                int min_neighbors,
                int flags,
                CvSize min_size);
bool cvZSetBreakOnError(bool value);
void cvZHeightRemap(CvMat* src, CvMat* dst, const float h0, const float m);
int cvZFindChessboardCorners(CvMat* img, CvSize patternSize,
                                CvPoint2D32fArrayAsFloats points,
                                JavaIntArray cornerCount,
                                int flags);
void cvZFindCornerSubPix(CvMat* img, CvPoint2D32fArrayAsFloats points,
                                CvSize win, CvSize zeroZone,
                                CvTermCriteria termCriteria);
void cvZDrawChessboardCorners(CvMat* img, CvSize patternSize,
                                CvPoint2D32fArrayAsFloats points,
                                int count, int wasFound);

typedef struct _CvZRNG { CvRNG val; } CvZRNG;
CvZRNG cvZCreateRNG (long seed);
void cvZRandArr(CvZRNG* rng, CvMat* arr,
                int dist_type, CvScalar param1, CvScalar param2);

%{
// Mahalanobis distance:
// takes the matrix x where each row is one vector, mu which is one-row matrix
// containing mean value vector, icovar which is inverse covariance matrix
// of the distribution.
//
// Fills dst_ matrix (same height as x, one column) with squares of Mahalanobis
// distance for each vector in x.

// the real work is done by the helper function below
template<typename T>
void calcMahalanobis2(const cv::Mat x, unsigned int dim, unsigned int rows,
        const cv::Mat mu, const cv::Mat icovar, CvMat* dst_)
// dim must be the dimension (the number of columns in x)
// rows bust be the number of data points (the number of rows in x)
{
    const T* mp = mu.ptr<T>(0);

    std::vector<T> centered;
    centered.resize(dim);

    unsigned int i, j;

    for (unsigned int row = 0; row < rows; row++) {

        const T* xp = x.ptr<T>(row);

        for (i = 0; i < dim; i++)
            centered[i] = xp[i] - mp[i];

        double res = 0;

        for (i = 0; i < dim; i++) {

            const T* ip = icovar.ptr<T>(i);
            double sum = 0;

            for (j = 0; j < dim; j++)
                sum += centered[j] * ip[j];

            res += sum * centered[i];
        }

        cvSetReal2D(dst_, row, 0, res);
    }
}

void cvZMahalanobis2(CvMat* x_, CvMat* mu_, CvMat* icovar_, CvMat* dst_)
{
    using namespace cv;
    CvMat tmp;
    cv::Mat x(cvGetMat(x_, &tmp));
    cv::Mat mu(cvGetMat(mu_, &tmp));
    cv::Mat icovar(cvGetMat(icovar_, &tmp));

    int type = x.type();

    // all images should be of the same type
    CV_Assert(mu.type() == type);
    CV_Assert(icovar.type() == type);
    CV_Assert(cvGetElemType(dst_) == type);

    Size sz = x.size();
    int rows = sz.height;
    int dim = sz.width;

    Size msz = mu.size();
    Size isz = icovar.size();

    // mu must be of the right dimension
    CV_Assert(msz.height == 1);
    CV_Assert(msz.width == dim);

    // icovar must be a square matrix of the right size
    CV_Assert(isz.height == isz.width);
    CV_Assert(isz.height == dim);

    if (type == CV_32FC1) {
        calcMahalanobis2<float>(x, dim, rows, mu, icovar, dst_);
    } else if (type == CV_64FC1) {
        calcMahalanobis2<double>(x, dim, rows, mu, icovar, dst_);
    } else {
        throw cv::Exception(CV_StsUnsupportedFormat,
                "cvZMahalanobis2 supports only CV_32FC1 or CV_64FC1 images.",
                "cvZMahalanobis2",
                __FILE__, __LINE__);
    }

}

void cvZConvexHull2(CvMat* points, CvMat* hull, int clockwise)
{
  cvConvexHull2(points, hull, clockwise, 0);
}

jobjectArray cvExtractSURFSimple(CvArr* arg1, CvArr* arg2,
                    CvSURFParams* arg3) {
    JNIEnv* jenv = JNU_GetEnv();
    CvMemStorage* storage = cvCreateMemStorage(0);
    // printf("cvCreateMemStorage %x\n", storage);
    CvSeq* pts;
    CvSeq* descrs;
    cvExtractSURF(arg1, arg2,
        &pts, &descrs,
        storage,
        *arg3);

    jfloatArray ptsArray =
        jenv->NewFloatArray(N_SURF_RECORD * pts->total);

    int BYTES_PER_FLOAT = 4;
    int dElem = descrs->elem_size / BYTES_PER_FLOAT;

    jfloatArray descrArray =
        jenv->NewFloatArray(dElem * descrs->total);

    jfloat* ptsPtr =
        (jfloat*)jenv->GetPrimitiveArrayCritical(ptsArray, 0);
    jfloat* descrPtr =
        (jfloat*)jenv->GetPrimitiveArrayCritical(descrArray, 0);

    for (int i = 0; i < pts->total; i++)
    {
        CvSURFPoint* pt = (CvSURFPoint*)cvGetSeqElem(pts, i);
        float* descr = (float*)cvGetSeqElem(descrs, i);

        int offs = i * N_SURF_RECORD;

        ptsPtr[offs + SURF_RECORD_X] = pt->pt.x;
        ptsPtr[offs + SURF_RECORD_Y] = pt->pt.y;
        ptsPtr[offs + SURF_RECORD_LAPLACIAN] = pt->laplacian;
        ptsPtr[offs + SURF_RECORD_SIZE] = pt->size;
        ptsPtr[offs + SURF_RECORD_DIR] = pt->dir;
        ptsPtr[offs + SURF_RECORD_HESSIAN] = pt->hessian;

        int dOffs = dElem * i;

        for (int j = 0; j < dElem; j++)
            descrPtr[dOffs + j] = descr[j];
    }

    // printf("cvReleaseMemStorage %x\n", storage);
    cvReleaseMemStorage(&storage);

    jenv->ReleasePrimitiveArrayCritical(ptsArray, ptsPtr, 0);
    jenv->ReleasePrimitiveArrayCritical(descrArray, descrPtr, 0);

    jobjectArray res = jenv->NewObjectArray(2,
                    jenv->FindClass("java/lang/Object"), 0);

    jenv->SetObjectArrayElement(res, 0, ptsArray);
    jenv->SetObjectArrayElement(res, 1, descrArray);

    return res;
}

jintArray cvHaarDetectObjectSimple(CvArr* arg1,
                CvHaarClassifierCascade* arg2,
                double arg3,
                int arg4,
                int arg5,
                CvSize arg6) {
    JNIEnv* jenv = JNU_GetEnv();
    CvMemStorage* storage = cvCreateMemStorage(0);
    // printf("cvCreateMemStorage %x\n", storage);
    CvSeq* res = cvHaarDetectObjects(arg1, arg2, storage,
                    arg3, arg4, arg5,
                    arg6);

    jintArray resArray =
        jenv->NewIntArray(N_HAAR_RECORD * res->total);

    jint* resPtr =
        (jint*)jenv->GetPrimitiveArrayCritical(resArray, 0);

    for (int i = 0; i < res->total; i++)
    {
        CvAvgComp* rec = (CvAvgComp*)cvGetSeqElem(res, i);

        int offs = i * N_HAAR_RECORD;

        resPtr[offs + HAAR_RECORD_X] = rec->rect.x;
        resPtr[offs + HAAR_RECORD_Y] = rec->rect.y;
        resPtr[offs + HAAR_RECORD_W] = rec->rect.width;
        resPtr[offs + HAAR_RECORD_H] = rec->rect.height;
        resPtr[offs + HAAR_RECORD_NEIGHBORS] = rec->neighbors;
    }

    // printf("cvReleaseMemStorage %x\n", storage);
    cvReleaseMemStorage(&storage);

    jenv->ReleasePrimitiveArrayCritical(resArray, resPtr, 0);

    return resArray;
}

bool cvZSetBreakOnError(bool value) {
    return cv::setBreakOnError(value);
}

// A bare-bones version of the height remap algorithm;
// the clojure wrapper in com.zenrobotics.titech.extract does
// some preprocessing work.
//
// See diagram: https://docs.google.com/a/zenrobotics.com/drawings/d/19D_E1WRRCWl9VMGf8AocnQFU6caXG0I0KZCuimbBY2c/edit
//
// src is the height map, offset so that baseline is 0
// dst will hold the remap coordinates; dst must be initialized to -1
// h0 is the height of the sensor, measured from the baseline
// m is the middle point (in pixels)
void cvZHeightRemap(CvMat* src, CvMat* dst, const float h0, const float m)
{
  CvSize size = cvGetSize(src);
  const int rows = size.height;
  const int cols = size.width;

  CvSize dst_size = cvGetSize(dst);
  CV_Assert(rows == dst_size.height);
  CV_Assert(cols == dst_size.width);

  // I don't know if it makes sense for m to be outside the range
  // [0, cols-1] but outside values occur easily when optimizing for
  // the parameters, and it seems useful to degrade gradually so an
  // optimizer can figure out which direction to go.
  const float m_clamped = std::min(std::max(m, 0.0f), (float) (cols-1));

  // See
  for (int j = 0; j < rows; j++) {
    float* srcr = (float *) (src->data.ptr + j * src->step);
    float* dstr = (float *) (dst->data.ptr + j * dst->step);
    float usedUpTo = m_clamped;
    // Sweep from the center to the beginning
    for (int i = (int) m_clamped; i >= 0; i--) {
      const float h1 = srcr[i];
      // Displacement from center pixel at current height
      const float x0 = i - m;
      // By equal triangles: extra displacement at baseline lvel
      const float x1 = (x0 * h1) / (h0 - h1);
      // Final position
      const float x = m + x0 + x1;

      // OK if not in the shadow of a higher object and height level
      // is not obviously broken
      if (h1 > -0.05 && x <= usedUpTo) {
        dstr[i] = x;
      }
      usedUpTo = (usedUpTo < x) ? usedUpTo : x;
    }
    // ... and to the end
    usedUpTo = m_clamped;
    for (int i = (int) m_clamped + 1; i < cols; i++) {
      const float h1 = srcr[i];
      const float x0 = i - m;
      const float x1 = (x0 * h1) / (h0 - h1);
      const float x = m + x0 + x1;
      if (h1 > -0.05 && x >= usedUpTo) {
        dstr[i] = x;
      }
      usedUpTo = (usedUpTo > x) ? usedUpTo : x;
    }
  }
}

int cvZFindChessboardCorners(CvArr* img, CvSize patternSize,
                                CvPoint2D32fArrayAsFloats points,
                                JavaIntArray cornerCount,
                                int flags)
{
    return cvFindChessboardCorners(
            img, patternSize,
            points.pointPtr,
            (int*)cornerCount.ptr,
            flags);
}

void cvZFindCornerSubPix(CvArr* img, CvPoint2D32fArrayAsFloats points,
                                CvSize win, CvSize zeroZone,
                                CvTermCriteria termCriteria)
{
    cvFindCornerSubPix(
            img,
            points.pointPtr,
            points.nPoints,
            win,
            zeroZone,
            termCriteria);
}

void cvZDrawChessboardCorners(CvArr* img, CvSize patternSize,
                                CvPoint2D32fArrayAsFloats points,
                                int count, int wasFound)
{
    cvDrawChessboardCorners(img, patternSize,
                points.pointPtr,
                count,
                wasFound);
}

typedef struct _CvZRNG { CvRNG val; } CvZRNG;

CvZRNG cvZCreateRNG (long seed)
{
  CvZRNG res;
  res.val = cvRNG(seed);
  return res;
}

void cvZRandArr(CvZRNG* rng, CvMat* arr,
                int dist_type, CvScalar param1, CvScalar param2)
{
  cvRandArr (&rng->val, arr, dist_type, param1, param2);
}

%}
