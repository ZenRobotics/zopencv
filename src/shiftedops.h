// Assuming CV includes already done etc.

// define shifted versions of some common operations
// that are used often and need top speed and minimum
// JNI/Java bookkeeping of anything.

// Two arguments, first argument is also destination, 
// second is shifted.
#define MAKE_SHIFTED_OP_2(name, operation) \
    void name(CvMat* src1dst, CvMat* src2, int dx, int dy, float invalid) \
    { \
        CvMat a, b;  \
        if (prepShifted(src1dst, src2, dx, dy, &a, &b)) \
          operation(&a, &b); \
        finishShifted(src1dst, dx, dy, invalid); \
    }

bool prepShifted(CvMat* src1dst, CvMat* src2, int dx, int dy, CvMat* a, CvMat* b)
{
    using namespace std;
    CvSize s = cvGetSize(src1dst);
    int w = s.width;
    int h = s.height;
    int pdx = max(dx, 0);
    int pdy = max(dy, 0);
    int ndx = max(-dx, 0);
    int ndy = max(-dy, 0);

    int rw = w - abs(dx);
    int rh = h - abs(dy);

    if (rw <= 0 || rh <= 0)
        return false;

    // Prepare output areas
    cvGetSubRect(src1dst, a, cvRect(pdx, pdy, rw, rh));
    cvGetSubRect(src2, b, cvRect(ndx, ndy, rw, rh));

    return true;
}

void finishShifted(CvMat* src1dst, int dx, int dy, float invalid)
{
    using namespace std;
    CvSize s = cvGetSize(src1dst);
    int w = s.width;
    int h = s.height;
    int pdx = max(dx, 0);
    int pdy = max(dy, 0);
    int ndx = max(-dx, 0);
    int ndy = max(-dy, 0);

    int rw = w - abs(dx);
    int rh = h - abs(dy);

    CvScalar v = cvScalar(invalid);
    // Fill area outside valid outputs
    if (pdx > 0)
        cvRectangle(src1dst,
                     cvPoint(0, 0),
                     cvPoint(pdx - 1, h),
                     v, -1, 8, 0);

    if (pdy > 0)
        cvRectangle(src1dst,
                     cvPoint(0, 0),
                     cvPoint(w, pdy - 1),
                     v, -1, 8, 0);
 
    if (ndx > 0)
        cvRectangle(src1dst, 
                     cvPoint(w - ndx, 0),
                     cvPoint(w - 1, h),
                     v, -1, 8, 0);

    if (ndy > 0)
        cvRectangle(src1dst, 
                     cvPoint(0, h - ndy),
                     cvPoint(w, h - 1),
                     v, -1, 8, 0);

}

inline void zadd(CvMat* a, CvMat* b) { cvAdd(a, b, a, NULL); }
inline void zmul(CvMat* a, CvMat* b) { cvMul(a, b, a); }
inline void zmin(CvMat* a, CvMat* b) { cvMin(a, b, a); }
inline void zmax(CvMat* a, CvMat* b) { cvMax(a, b, a); }

MAKE_SHIFTED_OP_2(cvZShiftedAdd, zadd)
MAKE_SHIFTED_OP_2(cvZShiftedMul, zmul)
MAKE_SHIFTED_OP_2(cvZShiftedMax, zmax)
MAKE_SHIFTED_OP_2(cvZShiftedMin, zmin)
