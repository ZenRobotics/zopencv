// Segmenting images and handling polygons

jint cvZSegmentIntoObjects(CvArr* image,
                           JavaFloatArray xs,
                           JavaFloatArray ys,
                           JavaFloatArray as,
                           JavaFloatArray m11s,
                           JavaFloatArray m20s,
                           JavaFloatArray m02s);
jint cvZColorConnectedComponents(CvMat* image, CvMat* ids, int chainMethod, int
                                 fillConnection);
jint cvZColorConnectedComponents2 (CvMat* _mask, CvMat* _label);
jobjectArray cvZComponentExtents (int max_id, CvMat* mat);

void cvZFillConvexPoly(CvMat* img, CvPointArrayAsInts points,
                       CvScalar color, int line_type, int shift);

void cvZFillOnePoly(CvMat* img, CvPointArrayAsInts points,
                    CvScalar color, int line_type, int shift);

void cvZFillRotatedRectangle(CvMat* img, float cx, float cy,
                             float width, float height, float angle,
                             CvScalar color, int line_type);
void cvZScatterSum(CvMat* ids_, CvMat* vals_, CvMat* dst_);
void cvZGather(CvMat* ids_, CvMat* vals_, CvMat* dst_);
void cvZScatterMoments(CvMat* ids, CvMat* dst);

%ignore zmin;
%ignore zmax;

%{

// Segment the single-channel image into objects using the given
// temporary storage. For object #i, set (xs[i], ys[i]) to be its
// centroid coordinates, as[i] its area, and m11s, m20s and m02s
// to the second-degree uncentralized spatial moments. Return the
// number of objects found.
//
// NB: You will want to add a black frame around image using
//   e.g. cvCopyMakeBorder
// There are two reasons for this:
// 1) http://tech.groups.yahoo.com/group/OpenCV/message/23486
//    cvFindContours fails to find blobs that touch the image edge,
//    which is "just a feature of [the] algorithm". The recommended
//    solution is to create a 1-pixel border around the image.
// 2) cvFindContours may modify the input image, so making a copy is
//    necessary if we have any further use for the image.
//
// After adding the frame, remember that coordinates are offset by
// +1, +1.

// Because Visual Studio doesn't do std::min and std::max
static int zmin(int a, int b) { return (a < b) ? a : b; }
static int zmax(int a, int b) { return (a > b) ? a : b; }

jint cvZSegmentIntoObjects(CvArr* image,
                           JavaFloatArray xs,
                           JavaFloatArray ys,
                           JavaFloatArray as,
                           JavaFloatArray m11s,
                           JavaFloatArray m20s,
                           JavaFloatArray m02s)
{
    CvSeq* firstContour = NULL;
    CvMemStorage* storage = cvCreateMemStorage(0);
    unsigned i = 0;

    cvFindContours(image, storage, &firstContour, sizeof(CvContour),
                   CV_RETR_EXTERNAL);

    if (firstContour != NULL)
    {
        unsigned arrayLen = xs.size;
        CV_Assert(ys.size == arrayLen && as.size == arrayLen &&
                  m11s.size == arrayLen && m20s.size == arrayLen &&
                  m02s.size == arrayLen);

        for(CvSeq* c=firstContour;
            c != NULL && i < arrayLen;
            c = c->h_next)
            {
                CvMoments moments;
                cvMoments(c, &moments, 1);
                float m00, m10, m01;

                // Find centroid using moments
                m00 = cvGetSpatialMoment(&moments, 0, 0);
                m10 = cvGetSpatialMoment(&moments, 1, 0);
                m01 = cvGetSpatialMoment(&moments, 0, 1);

                if (m00 < 1.0)   // OpenCV bug? polygon of zero area
                    continue;

                xs.ptr[i] = m10/m00;
                ys.ptr[i] = m01/m00;
                as.ptr[i] = m00;
                m11s.ptr[i] = cvGetSpatialMoment(&moments, 1, 1);
                m20s.ptr[i] = cvGetSpatialMoment(&moments, 2, 0);
                m02s.ptr[i] = cvGetSpatialMoment(&moments, 0, 2);
                i++;
            }
    }

    cvReleaseMemStorage(&storage);
    return i;
}

// From a binary image in "image", find the
// connected components and mark each location in ids
// with the id of the component it belongs to.
//
// Component IDs start at 1, and the rest is zero.
// Returns total number of ids, including 0 for ground
// (i.e., if there are two objects, their ids will be 1 and 2,
// the id of ground will be 0, and the function will return 3).
#define Z_COMP_ID_TYPE CV_16UC1
jint cvZColorConnectedComponents(CvMat* image, CvMat* ids, int chainMethod, int
                fillConnection)
{
    CV_Assert(cvGetElemType(ids) == Z_COMP_ID_TYPE);

    CvMemStorage* mem = cvCreateMemStorage(0);
    CvSeq* contours;

    cvFindContours(image, mem, &contours, sizeof(CvContour),
      CV_RETR_CCOMP, chainMethod);

    cvSetZero(ids);

    int id = 1;
    for (CvSeq* contour = contours; contour; contour = contour->h_next)
    {
      CV_Assert(id <= 0xFFFF); // XXX -- or just bail out?
      CvScalar color = cvScalar(id);
      cvDrawContours(ids, contour, color, color, -1, CV_FILLED,
                      fillConnection);
      id++;
    }

    cvReleaseMemStorage(&mem);

    return id;
}

// cvZColorConnectedComponents2 labels 4-connected foreground (> 0)
// components in mask and returns the labels in image label.
// Mask must be CV_8UC1 and labels CV16_UC1 with the same size as mask.
// Returns n+1, where n is the number of 4-connected components found.
// Upon exit, labels has each pixel corresponding to a background pixel in
// mask set to 0 and each pixel corresponding to a foreground pixel in mask
// set to the label id of a 4-connected component in mask.
//
// The reason for this routine is that the original connected components
// routine above does not work as well as it should (apparently
// changes component boundaries after filling).
jint cvZColorConnectedComponents2 (CvMat* _mask, CvMat* _label)
{
    using namespace cv;
    CV_Assert(cvGetElemType(_mask) == CV_8UC1);
    CV_Assert(cvGetElemType(_label) == Z_COMP_ID_TYPE);
    Mat1b mask  (_mask);
    Mat1w label (_label);
    CV_Assert(mask.size() == label.size());
    const unsigned int h = mask.rows;
    const unsigned int w = mask.cols;

    // Disjoint set forest for label sets implemented in an array
    // of two unsigneds: Parent and rank.
    struct
    {
        // Create a new label id.
        unsigned int New ()
        {
            unsigned int id = sets.size();
            sets.push_back (std::pair<unsigned int,unsigned int>(id, 0));
            return id;
        }

        // Find representative of set containing label id.
        unsigned int Find (unsigned int id)
        {
            unsigned int i;
            CV_Assert(id >= 0 && id < sets.size());
            // Find set representative by following parent links.
            for (i = id; Parent(i) != i; i = Parent(i))
                continue;
            // Apply path compression.
            for (unsigned int j = id; j != i;)
            {
                unsigned int p = Parent(j);
                Parent(j) = i;
                j = p;
            }
            return i;
        }

        // Fuse label sets containing ids _i and _j.
        unsigned int Fuse (unsigned int _i, unsigned int _j)
        {
            CV_Assert(_i >= 0 && _i < sets.size());
            CV_Assert(_j >= 0 && _j < sets.size());
            unsigned int i = Find (_i);
            unsigned int j = Find (_j);
            CV_Assert(i >= 0 && i < sets.size());
            CV_Assert(j >= 0 && j < sets.size());
            // Union by rank.
            unsigned int ij;
            if (Rank(i) < Rank(j))
                ij = Parent(i) = j;
            else if (Rank(i) > Rank(j))
                ij = Parent(j) = i;
            else
            {
                ij = Parent(i) = j;
                ++Rank(j);
            }
            return ij;
        }

        // Index of parent for set i in sets.
        // Parent(i) == i for representative elements.
        unsigned int& Parent (unsigned int i) { return sets[i].first; }

        // Rank of labelset i. (Used only for representative elements).
        unsigned int& Rank (unsigned int i) { return sets[i].second; }

        std::vector< std::pair<unsigned int,unsigned int> > sets;

    } labels;

    // Connected components labeling.
    //
    // The goal is to identify all 4-connected components in mask.
    // The first pass assigns an interim label to each foreground pixel in
    // mask. The labeling goes through the image row by row creating new
    // labels when it cannot reuse an old label from the left or above
    // neighbor of the current pixel and fuses labels when two labels meet.
    // The disjoint set forest above keeps track of fused labels.
    // Each set in the forest corresponds to a 4-connected component in
    // mask found so far. The separate stages below handle the cases when
    // a neighbor of a pixel is missing (the first pixel, the first row,
    // each row's first pixel). Finally each set (4-connected component) in
    // the forest is relabeled with an unique id assigned for that set.

    // Label 0 is reserved for background.
    labels.New();

    // Label first pixel.
    if (mask (0, 0))
        label(0, 0) = labels.New();
    else
        label(0, 0) = 0;

    // Label first row. Extend label from left if labeled.
    for (unsigned int x = 1; x < w; ++x)
        if (mask (0, x))
            label(0, x) = mask(0, x-1) ? label(0, x-1) : labels.New();
        else
            label(0, x) = 0;

    // Label subsequent rows.
    for (unsigned int y = 1; y < h; ++y)
    {
        // Handle first pixel. Extend label from above if labeled.
        if (mask(y, 0))
            label(y, 0) = mask(y-1, 0) ? label(y-1, 0) : labels.New();
        else
            label(y, 0) = 0;

        // Rest of the row. Extend label from left or above. Fuse labels
        // if both left and above labeled.
        for (unsigned int x = 1; x < w; ++x)
        {
            if (mask (y, x))
            {
                if (mask(y, x-1) && mask(y-1, x))
                    label(y, x) = labels.Fuse (label(y, x-1), label(y-1, x));
                else if (mask(y, x-1))
                    label(y, x) = label(y, x-1);
                else if (mask(y-1, x))
                    label(y, x) = label(y-1, x);
                else
                    label(y, x) = labels.New ();
            }
            else
            {
                label(y, x) = 0;
            }
        }
    }

    // Relabel segments. Assign a set id to each set of labels
    // contiguously from 1 to labels-1. Set id 0 is background.
    // Precomputes a relabeling array for each label used.
    int nlabels = 0;
    std::vector<unsigned int> relabel (labels.sets.size(), 0);
    for (unsigned int i = 0; i < labels.sets.size(); ++i)
    {
        unsigned int p = labels.Find (i);
        unsigned int l = relabel[p];
        if (l == 0)
            l = relabel[p] = nlabels++;
        relabel[i] = l;
    }

    // Relabel image. Overwrite image containing interim labels (there
    // are multiple interim labels per 4-connected component) with the
    // corresponding 4-connected component label.
    for (unsigned int y = 0; y < h; ++y)
        for (unsigned int x = 0; x < w; ++x)
            label(y,x) = relabel[label(y,x)];

    return nlabels;
}

// Find component extents from colored CV_16UC1 matrix.
// Returns arrays xmin ymax ymin ymax containing bounding
// rectangles for components.
// max_id is the largest object id so there are max_id+1
// possible values in mat - zero is background, 1 to max_id are objects

jobjectArray cvZComponentExtents (int max_id, CvMat* mat)
{
    using namespace cv;
    const int n_ids = max_id + 1;
    Mat img (mat);
    CV_Assert(img.type() == DataType<unsigned short>::type);

    JNIEnv* jenv = JNU_GetEnv();
    jobjectArray result = jenv->NewObjectArray (4,
                                                jenv->FindClass("[I"),
                                                NULL);
    jintArray result_min_x  = jenv->NewIntArray (n_ids);
    jintArray result_max_x  = jenv->NewIntArray (n_ids);
    jintArray result_min_y  = jenv->NewIntArray (n_ids);
    jintArray result_max_y  = jenv->NewIntArray (n_ids);
    jenv->SetObjectArrayElement (result, 0, result_min_x);
    jenv->SetObjectArrayElement (result, 1, result_max_x);
    jenv->SetObjectArrayElement (result, 2, result_min_y);
    jenv->SetObjectArrayElement (result, 3, result_max_y);
    jint* min_x = (jint*)jenv->GetPrimitiveArrayCritical(result_min_x, 0);
    jint* max_x = (jint*)jenv->GetPrimitiveArrayCritical(result_max_x, 0);
    jint* min_y = (jint*)jenv->GetPrimitiveArrayCritical(result_min_y, 0);
    jint* max_y = (jint*)jenv->GetPrimitiveArrayCritical(result_max_y, 0);

    std::fill (min_x, min_x+n_ids, img.cols-1);
    std::fill (max_x, max_x+n_ids, 0);
    std::fill (min_y, min_y+n_ids, img.rows-1);
    std::fill (max_y, max_y+n_ids, 0);

    for (int y = 0; y < img.rows; ++y)
        for (int x = 0; x < img.cols; ++x) {
            unsigned short id = img.at<unsigned short>(y, x);
            min_x[id] = zmin (min_x[id], x);
            max_x[id] = zmax (max_x[id], x);
            min_y[id] = zmin (min_y[id], y);
            max_y[id] = zmax (max_y[id], y);
        }

    jenv->ReleasePrimitiveArrayCritical(result_min_x, min_x, 0);
    jenv->ReleasePrimitiveArrayCritical(result_max_x, max_x, 0);
    jenv->ReleasePrimitiveArrayCritical(result_min_y, min_y, 0);
    jenv->ReleasePrimitiveArrayCritical(result_max_y, max_y, 0);

    return result;
}

void cvZFillConvexPoly(CvArr* img, CvPointArrayAsInts points,
                       CvScalar color, int line_type, int shift)
{
    cvFillConvexPoly(img,
                     points.pointPtr,
                     points.nPoints,
                     color,
                     line_type,
                     shift);
}

void cvZFillOnePoly(CvArr* img, CvPointArrayAsInts points,
                    CvScalar color, int line_type, int shift)
{
    cvFillPoly(img,
               &points.pointPtr,
               &points.nPoints,
               1,
               color,
               line_type,
               shift);
}

void cvZFillRotatedRectangle(CvMat* img, float cx, float cy,
                             float width, float height, float angle,
                             CvScalar color, int line_type)
{
    cv::Mat mat(img);
    cv::Point2f center(cx, cy);
    cv::Size2f size(width, height);
    cv::RotatedRect rect(center, size, angle);
    cv::Point2f vertices[4];
    rect.points(vertices);
    // cvFillConvexPoly requires integer vertices; do rounding
    cv::Point intVertices[4];
    for (int i = 0; i < 4; i++)
        intVertices[i] = vertices[i];
    cv::fillConvexPoly(mat,
                       intVertices,
                       4,
                       color,
                       line_type,
                       0);
}

namespace {
using namespace cv;
template <typename T>
void cvZScatterT(Mat1w ids, CvMat* dst_, CvMat* vals_)
{
    T vals(vals_);
    T dst(dst_);

    for (int y = 0; y < ids.rows; y++)
        for (int x = 0; x < ids.cols; x++)
        {
            CV_Assert(ids(y, x) < dst.cols);
            dst(0, ids(y, x)) += vals(y, x);
        }
}
} // namespace

// Some scatter-gather primitives expected to be useful.
// TODO: more efficient implementations using various
// techniques, there are just plain C (though for many cases,
// even that is fast enough)

// Calculate separate sums over each component; returns a single-row
// matrix.
// loop(x,y): dst(ids(x,y),0) += vals(x,y)
// Note: indexing in x,y notation instead of OpenCV row,col notation.
void cvZScatterSum(CvMat* ids_, CvMat* vals_, CvMat* dst_)
{
    int elemType = cvGetElemType(vals_);
    CV_Assert(cvGetElemType(ids_) == Z_COMP_ID_TYPE);
    CV_Assert(cvGetElemType(dst_) == elemType);

    using namespace cv;
    Mat1w ids(ids_);

    switch(elemType)
    {
    case CV_32FC1:
        cvZScatterT<Mat1f>(ids, dst_, vals_);
        break;
    case CV_64FC1:
        cvZScatterT<Mat1d>(ids, dst_, vals_);
        break;
    default:
        CV_Assert(false && (char*) "Unknown image type in cvZScatterSum");
    }
}

namespace {
using namespace cv;
// Implementation of cvZScatterMoments for destination matrices of type T
template <typename T>
void cvZScatterMomentsT(Mat1w ids, CvMat* dst_)
{
    T dst(dst_);
    int dst_rows = dst.rows;
    CV_Assert(dst.cols == 6);

    for (int y = 0; y < ids.rows; y++) {
        double y2 = y * y;
        for (int x = 0; x < ids.cols; x++) {
            double x2 = x * x;
            double xy = x * y;
            int row = ids(y, x);
            CV_Assert(row < dst_rows);
            dst(row, 0) += 1.0;
            dst(row, 1) += x;
            dst(row, 2) += y;
            dst(row, 3) += x2;
            dst(row, 4) += xy;
            dst(row, 5) += y2;
        }
    }
}
} // namespace

// Helper for zopencv-clj/cvZMultipleSpatialMoments*
// ids_: object-ids image
// dst: image of size (n-ids x 6)
// Writes moments m00, m10, m01, m20, m11, m02 into dst.
void cvZScatterMoments(CvMat* ids_, CvMat* dst)
{
    int elemType = cvGetElemType(dst);
    CV_Assert(cvGetElemType(ids_) == Z_COMP_ID_TYPE);

    using namespace cv;
    Mat1w ids(ids_);
    switch(elemType)
    {
    case CV_32FC1:
        cvZScatterMomentsT<Mat1f>(ids, dst);
        break;
    case CV_64FC1:
        cvZScatterMomentsT<Mat1d>(ids, dst);
        break;
    default:
        CV_Assert(false && (char*) "Unknown image type in cvZScatterMoments");
    }
}

// ids: An MxN matrix with identifiers (0-K), type CV_16UC1
// vals: A float row-vector (1xK) with values corresponding to identifiers
// dst: MxN float matrix, with each element having the value
//      corresponding to the identifier in the same place in the ids matrix.
// or put shortly: dst(x,y) = vals(ids(x,y), 0)
void cvZGather(CvMat* ids_, CvMat* vals_, CvMat* dst_)
{
    CV_Assert(cvGetElemType(ids_) == Z_COMP_ID_TYPE);
    CV_Assert(cvGetElemType(vals_) == CV_32FC1);
    CV_Assert(cvGetElemType(dst_) == CV_32FC1);

    using namespace cv;
    Mat1w ids(ids_);
    Mat1f vals(vals_);
    Mat1f dst(dst_);

    for (int y = 0; y < ids.rows; y++)
    {
        for (int x = 0; x < ids.cols; x++)
        {
            dst(y, x) = vals(0, ids(y, x));
        }
    }
}

%}
