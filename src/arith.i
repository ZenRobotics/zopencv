// Arithmetic & linear algebra

/* Misc */
CvMat* cvZGetPerspectiveTransform(CvPoint2D32fArrayAsFloats src,
                                  CvPoint2D32fArrayAsFloats src,
                                  CvMat *);
CvMat* cvZGetAffineTransform(CvPoint2D32fArrayAsFloats src,
                                  CvPoint2D32fArrayAsFloats src,
                                  CvMat *);
void cvZVecOuterProduct(const CvMat* src, CvMat* dst);
void cvZCopy(const CvMat* src, CvMat* dst);
void cvZWriteRow(CvMat* dst, int index, JavaFloatArray row);
void cvZAddRow(CvMat* dst, int index, JavaFloatArray row);

void cvZAvg(CvMat* img, JavaDoubleArray out, int index);
void cvZAdd(CvMat* src1, CvMat* src2, CvMat* dst);
void cvZScale(CvMat* src,
        JavaDoubleArray scale, int scaleindex,
        JavaDoubleArray shift, int shiftindex,
        CvMat* dst);
void cvZMul(CvMat* src1, CvMat* src2, CvMat* dst);
void cvZMatMul(CvMat* src1, CvMat* src2, CvMat* dst);
void cvZDiv(CvMat* src1, CvMat* src2, CvMat* dst);
void cvZMinS(CvMat* src, JavaDoubleArray value, int index, CvMat* dst);
void cvZMaxS(CvMat* src, JavaDoubleArray value, int index, CvMat* dst);
void cvZMinMaxLoc(CvMat* src, JavaFloatArray minmax,
                CvPoint* minLoc, CvPoint* maxLoc, CvMat* mask);
void cvZThreshold1(CvMat* src, CvMat* dst, JavaDoubleArray threshold, int idxT,
                   JavaDoubleArray maxValue, int idxM, int mode);
void cvZCalcCovarMatrix(const CvMat** vects, int count,
                        CvMat* cov_mat, CvMat* avg, int flags);

/* Maximums, minimums and zeros */
int cvZCumulativeExceeds(CvMat* data_, float threshold);
void cvZGridLocalMaxima(CvMat* arg_, int elementSize, JavaIntArray x, JavaIntArray y, JavaFloatArray v);
jint cvZMaxZeroRun(CvMat* mat, JavaIntArray ind);
jobjectArray cvZLocalMaxima(CvMat* img, float floor);
jobjectArray cvZGetNonzero(CvMat* mat);

/* SVD */
void cvZSVD(CvMat* A, CvMat* W, CvMat* U, int flags);
void cvZSVDArr(JavaDoubleArray a, int rows, int cols,
               JavaDoubleArray w,  // 1D
               JavaDoubleArray u,
               JavaDoubleArray v);

/* Histograms */
CvHistogram *cvZCreateHist(int size, float min, float max);
void cvZCalcHist(CvMat* src, CvHistogram* hist, int accumulate);
float cvZGetHistValue_1D(CvHistogram* hist, int index);

CvHistogram* cvZZCreateHist(int nDims, JavaIntArray sizes,
                                int tp, JavaFloatArray jranges,
                                int uniform);
void cvZZCalcHist(const CvMat** src, CvHistogram* hist, int accumulate, CvMat* mask);

void cvZZCalcBackProjectPatch(const CvMat** src, CvMat* dst, CvSize range, CvHistogram* hist, int method, double factor);
void cvZZCalcBackProject(const CvMat** src, CvMat* dst, CvHistogram* hist);

/* Moments */
void cvSpatialMoments(int binary, CvMat* img, JavaDoubleArray out, int index);
void cvZGetHuMoments(int binary, CvMat* img, JavaDoubleArray out);

%{

void cvZAvg(CvMat* img, JavaDoubleArray out, int index)
{
    CvScalar s = cvAvg(img, NULL);
    for (int i = 0; i < 4; i++)
        out.ptr[index + i] = s.val[i];
}

CvMat* cvZGetPerspectiveTransform(CvPoint2D32fArrayAsFloats src,
                                  CvPoint2D32fArrayAsFloats dst,
                                  CvMat * mat)
{
  return cvGetPerspectiveTransform(src.pointPtr, dst.pointPtr, mat);

}

CvMat* cvZGetAffineTransform(CvPoint2D32fArrayAsFloats src,
                             CvPoint2D32fArrayAsFloats dst,
                             CvMat * mat)
{
  return cvGetAffineTransform(src.pointPtr, dst.pointPtr, mat);
}

void cvZVecOuterProduct(const CvMat* src, CvMat* dst)
{
   cvMulTransposed(src, dst, 0, NULL, 1.0);
}

void cvZWriteRow(CvMat* dst, int index, JavaFloatArray row)
{
     CvSize siz = cvGetSize(dst);
     CV_Assert(row.size = siz.width);
     CV_Assert(cvGetElemType(dst) == CV_32FC1);

     CvMat cvrow;
     cvInitMatHeader(&cvrow, 1, row.size, CV_32FC1, row.ptr, 4 * row.size);

     CvMat target;
     cvGetSubRect(dst, &target, cvRect(0, index, row.size, 1));

     cvCopy(&cvrow, &target, NULL);
}

void cvZAddRow(CvMat* dst, int index, JavaFloatArray row)
{
     CvSize siz = cvGetSize(dst);
     CV_Assert(row.size = siz.width);
     CV_Assert(cvGetElemType(dst) == CV_32FC1);

     CvMat *cvrow = cvCreateMatHeader(1, row.size, CV_32FC1);
     cvSetData(cvrow, row.ptr, 4*row.size);

     CvMat target;
     cvGetSubRect(dst, &target, cvRect(0, index, row.size, 1));

     cvAdd(cvrow, &target, cvrow);

     cvCopy(cvrow, &target, NULL);
     cvReleaseMat(&cvrow);
}

void cvZAdd(CvMat* src1, CvMat* src2, CvMat* dst)
{
    cvAdd(src1,src2,dst,NULL);
}

void cvZMul(CvMat* src1, CvMat* src2, CvMat* dst)
{
    cvMul(src1,src2,dst,1);
}

void cvZDiv(CvMat* src1, CvMat* src2, CvMat* dst)
{
    cvDiv(src1,src2,dst,1);
}

void cvZMatMul(CvMat* src1, CvMat* src2, CvMat* dst)
{
    cvMatMul(src1, src2, dst);
}

void cvZMinS(CvMat* src, JavaDoubleArray value, int index, CvMat* dst)
{
    cvMinS(src, value.ptr[index], dst);
}

void cvZMaxS(CvMat* src, JavaDoubleArray value, int index, CvMat* dst)
{
    cvMaxS(src, value.ptr[index], dst);
}

void cvZMinMaxLoc(CvMat* src, JavaFloatArray minmax,
                CvPoint* minLoc, CvPoint* maxLoc, CvMat* mask)
{
    double tmin, tmax;
    cvMinMaxLoc(src, &tmin, &tmax, minLoc, maxLoc, mask);
    CV_Assert(minmax.size >= 2);
    minmax.ptr[0] = tmin;
    minmax.ptr[1] = tmax;
}

void cvZScale(CvMat* src,
        JavaDoubleArray scale, int scaleindex,
        JavaDoubleArray shift, int shiftindex,
        CvMat* dst)
{
        cvScale(src, dst,
                (double) scale.ptr[scaleindex],
                (double) shift.ptr[shiftindex]);
}

void cvZThreshold1(CvMat* src, CvMat* dst,
                   JavaDoubleArray threshold, int idxT,
                   JavaDoubleArray maxValue, int idxM,
                   int mode)
{
    cvThreshold(src, dst, threshold.ptr[idxT], maxValue.ptr[idxM], mode);
}

// Wrapper needed for double pointer cast
void cvZCalcCovarMatrix(const CvMat** vects, int count,
                CvMat* cov_mat, CvMat* avg, int flags)
{
    cvCalcCovarMatrix((const CvArr**)vects, count, cov_mat, avg, flags);
}


// Return the smallest index such that the cumulative sum of data_
// first exceeds threshold at that index. The matrix data_ is indexed
// and the cumulative sum performed in some unspecified order that
// coincides with the natural order for both row and column vectors.
// Useful for finding quantiles: if data_ sums to 1, the median is
// found with threshold 0.5. If threshold exceeds the sum of data_,
// will return one past the end of data_.
int cvZCumulativeExceeds(CvMat* data_, float threshold)
{
    using namespace cv;
    Mat data(data_);
    CV_Assert(data.type() == DataType<float>::type);
    double sum = 0.0;

    MatConstIterator_<float> it = data.begin<float>();
    MatConstIterator_<float> it_end = data.end<float>();
    double accumulator = 0.0;
    int i = 0;

    for (; it != it_end; it++, i++) {
        accumulator += *it;
        if (accumulator >= threshold) {
            return i;
        }
    }
    return i;
}

/** Find the location of the local maxima inside each grid element
 * Semantics: arg is divided into elementSize*elementSize pixel subimages
 * starting from (0,0) (i.e., if not evenly divided,
 * the last grid elements are left smaller).
 *
 * Inside each grid element, a search for the local maximum is carried
 * out using minMaxLoc. The returns are combined into the arrays
 * x, y (in global image coordinates) and v (the value at the maximum).
 *
 * The input arrays should have size
 * ceil(arg.cols / elementSize) * ceil(arg.rows / elementSize), and
 * all entries will be filled.
 */
void cvZGridLocalMaxima(CvMat* arg_, int elementSize, JavaIntArray x, JavaIntArray y, JavaFloatArray v)
{
    using namespace cv;
    Mat arg(arg_);
    CV_Assert(x.size == y.size);
    CV_Assert(x.size == v.size);
    unsigned int idx = 0;
    for (int gy = 0; gy < arg.rows; gy += elementSize)
    {
        int gyend = min(gy + elementSize, arg.rows);
        for (int gx = 0; gx < arg.cols; gx += elementSize)
        {
            int gxend = min(gx + elementSize, arg.cols);

            Mat g = arg(Range(gy, gyend), Range(gx, gxend));

            double maxVal;
            Point maxLoc;
            minMaxLoc(g, NULL, &maxVal, NULL, &maxLoc);
            CV_Assert(idx < x.size);
            x.ptr[idx] = maxLoc.x + gx;
            y.ptr[idx] = maxLoc.y + gy;
            v.ptr[idx] = maxVal;
            idx++;
        }
    }
    CV_Assert(idx == x.size);
}

// Return the maximal length of a run of zeros within a single-row
// CV_8UC1 matrix. If ind is not empty, it must be a two-element array
// and the beginning and end of the first maximal run are stored in it.
// In case of no run, the array gets values -1, -1.
jint cvZMaxZeroRun(CvMat* mat, JavaIntArray ind)
{
    CV_Assert(cvGetElemType(mat) == CV_8UC1);
    CV_Assert(mat->rows == 1);
    CV_Assert(ind.size == 0 || ind.size == 2);

    if (ind.size == 2)
    {
        ind.ptr[0] = -1;
        ind.ptr[1] = -1;
    }

    int currentRunStarted = 0;
    int maximumRun = 0;
    for (int i = 0; i < mat->cols; i++)
    {
        uchar* ptr = (mat->data.ptr + i);
        if (*ptr != 0)
        {
            currentRunStarted = i + 1;
        }
        else if (i - currentRunStarted + 1 > maximumRun)
        {
            maximumRun = i - currentRunStarted + 1;
            if (ind.size == 2)
            {
                ind.ptr[0] = currentRunStarted;
                ind.ptr[1] = i;
            }
        }
    }
    return maximumRun;
}

namespace {
// step is the image widthStep.
template <typename T>
inline int is_maximum(T *p, int step, int i, int j, int h, int w) {
  return ((i == 0 || j == 0 || p[-step - 1] <= *p) &&
          (i == 0 || p[-step] <= *p) &&
          (i == 0 || j == (w - 1) || p[-step + 1] <= *p) &&
          (j == 0 || p[-1]  <= *p) &&
          (j == (w - 1) || p[1] <= *p) &&
          (j == 0 || i == (h - 1) || p[step - 1] <= *p) &&
          (i == (h - 1) || p[step] <= *p) &&
          (i == (h - 1) || j == (w - 1) || p[step + 1] <= *p));
}

template <typename T>
jobjectArray cvZLocalMaximaT(JNIEnv* jenv, CvArr* arg1, T floor) {
    IplImage tmp;
    IplImage *img = cvGetImage(arg1, &tmp);

    CV_Assert(img->nChannels == 1);
    CV_Assert(img->width > 2);
    CV_Assert(img->height > 2);

    int i,j;
    int step = img->widthStep / sizeof(T);
    int numMaxima = 0;
    // Go through the data once to find out the length
    // XXX(lrasinen) Borders are ignored
    T *data = (T*)img->imageData;
    T *p;
    const int h = img->height;
    const int w = img->width;
    for (i = 0; i < h; i++) {
      for (j = 0; j < w; j++) {
        p = data + (i * step + j);
        if (floor <= *p && is_maximum(p, step, i, j, h, w)) numMaxima++;
      }
    }
    // everything works fine if numMaxima == 0

    jintArray xArray = jenv->NewIntArray(numMaxima);
    jint* xPtr =
      (jint*)jenv->GetPrimitiveArrayCritical(xArray, 0);

    jintArray yArray = jenv->NewIntArray(numMaxima);
    jint* yPtr =
      (jint*)jenv->GetPrimitiveArrayCritical(yArray, 0);

    jfloatArray vArray = jenv->NewFloatArray(numMaxima);
    jfloat* vPtr =
      (jfloat*)jenv->GetPrimitiveArrayCritical(vArray, 0);

    // Go through the data again and add the data to the table
    for (i = 0; i < h; i++) {
      for (j = 0; j < w; j++) {
        p = data + (i * step + j);
        if (floor <= *p && is_maximum(p, step, i, j, h, w)) {
          *xPtr++ = j;
          *yPtr++ = i;
          *vPtr++ = (jfloat)*p;
        }
      }
    }

    jenv->ReleasePrimitiveArrayCritical(xArray, xPtr, 0);
    jenv->ReleasePrimitiveArrayCritical(yArray, yPtr, 0);
    jenv->ReleasePrimitiveArrayCritical(vArray, vPtr, 0);

    jobjectArray res = jenv->NewObjectArray(3,
                           jenv->FindClass("java/lang/Object"), 0);

    jenv->SetObjectArrayElement(res, 0, xArray);
    jenv->SetObjectArrayElement(res, 1, yArray);
    jenv->SetObjectArrayElement(res, 2, vArray);

    return res;
}

}  // namespace

jobjectArray cvZLocalMaxima(CvArr* arg1, float arg2) {
    JNIEnv* jenv = JNU_GetEnv();
    IplImage tmp;
    IplImage *img = cvGetImage(arg1, &tmp);
    if (img->depth == IPL_DEPTH_32F) {
        return cvZLocalMaximaT<float>(jenv, arg1, arg2);
    } else if (img->depth == IPL_DEPTH_64F) {
        return cvZLocalMaximaT<double>(jenv, img, (double)arg2);
    } else if (img->depth == IPL_DEPTH_8U) {
        unsigned char floor = arg2;
        if (arg2 < 0) floor = 0;
        if (arg2 > 255) floor = 255;
        return cvZLocalMaximaT<unsigned char>(jenv, arg1, floor);
    } else {
        CV_Assert(false && (char*)
                            "Unknown image depth (should not happen)");
        return 0;
    }
}

// Return an array of arrays of integer: [rows columns]
// where the corresponding entries in rows and columns
// are the row and column of every nonzero cell in mat
template<typename T>
    jobjectArray cvZGetNonzeroT(CvMat* mat, int nNonzero)
{
    JNIEnv* jenv = JNU_GetEnv();
    jintArray rowArray = jenv->NewIntArray(nNonzero);
    jintArray colArray = jenv->NewIntArray(nNonzero);
    jint* rowPtr = (jint*) jenv->GetPrimitiveArrayCritical(rowArray, 0);
    jint* colPtr = (jint*) jenv->GetPrimitiveArrayCritical(colArray, 0);

    int outOffset = 0;
    for (int row = 0; row < mat->height; row++)
    {
        T* thisRow = (T*) (mat->data.ptr + row * mat->step);
        for (int col = 0; col < mat->width; col++)
        {
            if (thisRow[col] != 0)
            {
                rowPtr[outOffset] = row;
                colPtr[outOffset] = col;
                outOffset++;
            }
        }
    }
    jenv->ReleasePrimitiveArrayCritical(colArray, colPtr, 0);
    jenv->ReleasePrimitiveArrayCritical(rowArray, rowPtr, 0);
    CV_Assert(outOffset == nNonzero);

    jobjectArray result = jenv->NewObjectArray(2,
                                               jenv->FindClass("java/lang/Object"),
                                               0);
    jenv->SetObjectArrayElement(result, 0, rowArray);
    jenv->SetObjectArrayElement(result, 1, colArray);

    return result;
}

jobjectArray cvZGetNonzero(CvMat* mat)
{
    IplImage tmp;
    IplImage* img = cvGetImage(mat, &tmp);
    int nNonzero = cvCountNonZero(img);

    switch(img->depth)
    {
    case IPL_DEPTH_8U:
        return cvZGetNonzeroT<uchar>(mat, nNonzero);
    case IPL_DEPTH_32F:
        return cvZGetNonzeroT<float>(mat, nNonzero);
    default:
        CV_Assert(false && "cvZGetNonzero requires 8U or 32F image");
        return 0;
    }
}

void cvZSVD(CvArr* A, CvArr* W, CvArr* U, int flags)
{
    cvSVD(A, W, U, NULL, flags);
}

void cvZSVDArr(JavaDoubleArray a, int rows, int cols,
               JavaDoubleArray w,  // 1D
               JavaDoubleArray u,
               JavaDoubleArray v)
{
    CvMat mA;
    CvMat mW;
    CvMat mU;
    CvMat mV;
    int nRes = (rows < cols ? rows : cols);
//    printf("cvZSVDArr! (%d %d -> %d) l: %d %d %d %d\n\n",
//            rows, cols, nRes,
//            a.size, w.size, u.size, v.size);

    cvInitMatHeader(&mA, rows, cols, CV_64FC1, a.ptr);
    cvInitMatHeader(&mW, nRes, 1, CV_64FC1, w.ptr);
    cvInitMatHeader(&mU, rows, rows, CV_64FC1, u.ptr);
    cvInitMatHeader(&mV, cols, cols, CV_64FC1, v.ptr);

    cvSVD(&mA, &mW, &mU, &mV, 0);

//    printf("cvZSVDArr done!\n");
}

// Easy-to-use versions of histogram functions that only support
// a single dimension

CvHistogram *cvZCreateHist(int size, float min, float max)
{
     float range[] = {min,max};
     float* ranges[] = {range};
     return cvCreateHist(1, &size, CV_HIST_ARRAY, ranges, 1);
}

void cvZCalcHist(CvArr* src,
                 CvHistogram* hist,
                 int accumulate)
{
     IplImage img;
     IplImage* imgp = &img;
     cvGetImage(src, &img);
     cvCalcHist(&imgp, hist, accumulate, 0);
}

// A function to wrap the cvGetHistValue_1D macro
float cvZGetHistValue_1D(CvHistogram* hist, int index)
{
     return *cvGetHistValue_1D(hist, index);
}

CvHistogram* cvZZCreateHist(int nDims, JavaIntArray sizes,
                            int tp, JavaFloatArray jranges,
                            int uniform)
{
    float** ranges = (float**)alloca(sizeof(float*) * nDims);
    if (uniform)
    {
        for (int i = 0; i < nDims; i++)
            ranges[i] = jranges.ptr + 2 * i;
    } else
    {
        int tot = 0;
        for (int i = 0; i < nDims; i++)
        {
            ranges[i] = jranges.ptr + tot;
            tot += sizes.ptr[i] + 1; // Both low and high
        }
    }
    return cvCreateHist(nDims, (int*)sizes.ptr,
                        tp, ranges, uniform);
}

void cvZZCalcHist(const CvMat** src, CvHistogram* hist, int accumulate, CvArr* mask)
{
    int nDims = cvGetDims(hist->bins);
    IplImage* imgs = (IplImage*)alloca(sizeof(IplImage) * nDims);
    IplImage** pImgs = (IplImage**)alloca(sizeof(IplImage*) * nDims);
    for (int i = 0; i < nDims; i++)
    {
        pImgs[i] = cvGetImage(src[i], &imgs[i]);
    }
    cvCalcHist(pImgs, hist, accumulate, mask);
}

void cvZZCalcBackProject(const CvMat** src, CvArr* dst, CvHistogram* hist)
{
    int nDims = cvGetDims(hist->bins);
    IplImage* imgs = (IplImage*)alloca(sizeof(IplImage) * nDims);
    IplImage** pImgs = (IplImage**)alloca(sizeof(IplImage*) * nDims);
    for (int i = 0; i < nDims; i++)
    {
        pImgs[i] = cvGetImage(src[i], &imgs[i]);
    }
    cvCalcBackProject(pImgs, dst, hist);
}

void cvZZCalcBackProjectPatch(const CvMat** src, CvArr* dst, CvSize range, CvHistogram* hist, int method, double factor)
{
    int nDims = cvGetDims(hist->bins);
    IplImage* imgs = (IplImage*)alloca(sizeof(IplImage) * nDims);
    IplImage** pImgs = (IplImage**)alloca(sizeof(IplImage*) * nDims);
    for (int i = 0; i < nDims; i++)
    {
        pImgs[i] = cvGetImage(src[i], &imgs[i]);
    }
    cvCalcBackProjectPatch(pImgs, dst, range, hist, method, factor);
}

void cvSpatialMoments(int binary, CvArr* img, JavaDoubleArray out, int index)
{
    CvMoments moments = CvMoments();
    cvMoments(img, &moments, binary);
    int i=0;
    for (int x=0; x<4; x++)
        for (int y=0; x+y<4; y++)
            out.ptr[index + i++] = cvGetSpatialMoment(&moments, x, y);
}

// Helper for zopencv-clj/cvZHuMoments
void cvZGetHuMoments(int binary, CvArr* img, JavaDoubleArray out)
{
    CvMoments moments;
    CvHuMoments huMoments;
    cvMoments(img, &moments, binary);
    cvGetHuMoments(&moments, &huMoments);
    out.ptr[0] = huMoments.hu1;
    out.ptr[1] = huMoments.hu2;
    out.ptr[2] = huMoments.hu3;
    out.ptr[3] = huMoments.hu4;
    out.ptr[4] = huMoments.hu5;
    out.ptr[5] = huMoments.hu6;
    out.ptr[6] = huMoments.hu7;
}

%}
