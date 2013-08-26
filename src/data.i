// Functions for converting, creating, saving and loading CvMats
CvMat* cvCloneArr(CvMat* arr);
CvMat* cvMatFromData(int height, int width, int type, const char* BYTE);
CvMat* cvMatFromDataAndOffset(int height, int width, int type, const char* BYTE, int offset);
void cvWritePnm(CvMat* arr, const char* filename);
void cvZSaveImage(const char* filename, CvMat* arr, int jpeg_quality);
void cvZSaveImage(const char* filename, CvMat* arr);
void cvZSave(const char* filename, const CvMat* arr);
CvMat* cvZLoad(const char* filename);
int cvGetContinuousDataSize(CvMat* img);
JavaDirectByteBuffer cvZGetRawDataAsByteBuffer(CvMat* img);
void cvZScaleTo8U(CvMat* tgt, CvMat* src);
void cvConvertArrToIntArray(JavaIntArray bytes, CvMat* img);
void cvCopyRectangleToByteArray(JavaByteArray bytes, CvMat* img, size_t offset, CvRect* rectangle);
void cvCopyArrToByteArray(JavaByteArray bytes, CvMat* img, size_t offset);
void cvCopyByteArrayToArr(CvMat* img, JavaByteArray data, size_t offset);
void cvCopyArrToFloatArray(JavaFloatArray floats, CvMat* img, size_t offset);
void cvCopyArrToDoubleArray(JavaDoubleArray doubles, CvMat* img, size_t offset);
void cvZCopy(const CvMat* src, CvMat* dst);
CvMat* cvZQueryFrame(CvCapture* capt);
int cvZWriteFrame(CvVideoWriter* writer, CvMat* mat);
void cvZSetX(CvMat* mat);
void cvZSetY(CvMat* mat);

%{

/* Glue for cvCloneMat, which doesn't work on CvMat directly */
CvMat* cvCloneArr(CvMat* arr)
{
    CvMat tmp;
    CvMat* mat = cvGetMat(arr, &tmp, NULL, 0);
    return cvCloneMat(mat);
}

CvMat* cvMatFromData(int height, int width, int type, const char* data)
{
    CvMat m = cvMat(height, width, type, (void*)data);
    return cvCloneMat(&m);
}

CvMat* cvMatFromDataAndOffset(int height, int width, int type, const char* data, int offset)
{
    CvMat m = cvMat(height, width, type, (void*)(data + offset));
    return cvCloneMat(&m);
}

void cvWritePnm(CvMat* arr, const char* filename)
{
     uchar * data;
     CvSize s;
     cvGetRawData(arr, &data, NULL, &s);
     if (data != 0)
     {
         IplImage tmpImg;
         IplImage* img = cvGetImage(arr, &tmpImg);
         FILE * f = fopen(filename, "wb");
         if (f == 0)
         {
             perror("Could not open file to write\n");
             exit(1);
         }
         switch(img->nChannels)
         {
         case 1:
             fprintf(f,"P5 %d %d 255\n", s.width, s.height);
             fwrite(data, s.width * s.height, 1, f);
             break;
         case 3:
             fprintf(f,"P6 %d %d 255\n", s.width, s.height);
             fwrite(data, s.width * s.height * 3, 1, f);
             break;
         default:
             fprintf(stderr,
                     "Don't know how to write %d-channel pnm files\n",
                     img->nChannels);
         }
         fclose(f);
     }
}

// The documentation of cvSaveImage says it only works for images
// with one or three channels, so we ensure that there is the right
// number. It actually worked with four channels but output a png
// image that did not behave well.
void cvZSaveImage(const char* filename, CvMat* arr, int jpeg_quality)
{
    IplImage tmp;
    IplImage* input = cvGetImage(arr, &tmp);
    CV_Assert(input->depth == IPL_DEPTH_8U || input->depth == IPL_DEPTH_16U);
    CV_Assert(input->nChannels == 1 || input->nChannels == 3
           || input->nChannels == 4);
    int parameters[] = { CV_IMWRITE_JPEG_QUALITY, jpeg_quality, 0 };

    switch(input->nChannels)
    {
    case 1:
    case 3:
        cvSaveImage(filename, arr, parameters);
        break;
    case 4:
        {
            IplImage* output = cvCreateImage(cvGetSize(input), input->depth, 3);
            cvCvtColor(arr, output, CV_BGRA2BGR);
            cvSaveImage(filename, output, parameters);
            cvReleaseImage(&output);
        }
        break;
    default:
        CV_Assert(0);
    }
}

void cvZSaveImage(const char* filename, CvMat* arr)
{
    cvZSaveImage(filename, arr, 95);
}

// Simplified interface to cvSave
void cvZSave(const char* filename, const CvMat* arr)
{
    cvSave(filename, arr, NULL, NULL, cvAttrList(NULL, NULL));
}

// Converse of cvZSave
// NOTE: cannot be used to load non-images like this (need to
// allocate memory separately) but works for images according to
// OpenCV documentation
CvMat* cvZLoad(const char* filename)
{
    return (CvMat*) cvLoad(filename, NULL, NULL, NULL);
}

// This function computes the number of bytes needed for storing all
// the pixels in this image. It is useful for e.g. computing the
// number of bytes cvCopyArrToByteArray needs.
//
// The data as actually laid out in memory might use more bytes than
// this due to non-contiguity. cvCopyArrToByteArray copies the bytes
// intelligently, so this function is safe for allocating the array.
int cvGetContinuousDataSize(CvMat* img)
{
     cv::Mat m(img);
     return m.cols * m.rows * m.elemSize();
}

JavaDirectByteBuffer cvZGetRawDataAsByteBuffer(CvMat* img) {
    uchar* data;
    CvSize size;
    int step;
    CV_Assert(CV_IS_MAT_CONT(img->type));
    cvGetRawData(img, &data, &step, &size);
    JavaDirectByteBuffer ret = { cvGetContinuousDataSize(img), data };
    return ret;
}

void cvZScaleTo8U(CvMat* tgt, CvMat* src)
{
    // The compiler doesn't analyze the nested switch with CV_Assert
    // to know that colorConversion is initialized in all reachable
    // branches.
    int colorConversion = -1;
    int doColorConversion = 1;
    CvSize size = cvGetSize(src);
    int elemTypeSrc = cvGetElemType(src);
    int channelsSrc = CV_MAT_CN(elemTypeSrc);
    int channelsTgt = CV_MAT_CN(cvGetElemType(tgt));

    switch(channelsSrc) {
    case 1:
        switch(channelsTgt) {
        case 3:
            colorConversion = CV_GRAY2RGB;
            break;
        case 4:
            colorConversion = CV_GRAY2RGBA;
            break;
        default:
            CV_Assert(0);
        }
        break;
    case 3:
        switch(channelsTgt) {
        case 3:
            doColorConversion = 0;
            break;
        case 4:
            colorConversion = CV_RGB2RGBA;
            break;
        default:
            CV_Assert(0);
        }
        break;
    case 4:
        switch(channelsTgt) {
        case 3:
            colorConversion = CV_RGBA2RGB;
            break;
        case 4:
            doColorConversion = 0;
            break;
        }
        break;
    default:
        CV_Assert(0 && "Don't know how to handle those channels");
    }
    CV_Assert(!doColorConversion || colorConversion != -1);

    switch(cvGetElemType(src)) {
    case CV_8UC1:
    case CV_8UC3:
    case CV_8UC4:
        if (doColorConversion)
            cvCvtColor(src, tgt, colorConversion);
        else
            cvCopy(src, tgt, NULL);
      break;
    case CV_32FC1:
    case CV_16UC1:
      {
        // Convert single-channel floating point or 16-bit image into a
        // viewable 8-bit integer image
        CvMat* tmp = cvCreateMat(size.height, size.width, CV_8UC1);
        double tmin, tmax;
        cvMinMaxLoc(src, &tmin, &tmax, NULL, NULL, NULL);
        double diff = tmax - tmin;
        double scale = 255.0 / (diff + 0.00001);
        double offset = scale * (-tmin);
        cvConvertScale(src, tmp, scale, offset);
        if (doColorConversion)
            cvCvtColor(tmp, tgt, colorConversion);
        else
            cvCopy(tmp, tgt, NULL); // this branch is actually never reached now
        cvReleaseMat(&tmp);
      }
      break;
    case CV_32FC3:
      {
        // TODO(jks) if we ever really want to view these
        // it's nonobvious how exactly to handle the conversion
        // now we just guess that we want 0..1 to map to 0..255
        // XXX(jkaasine) except the code below maps -0.5..0.5
        CvMat* tmp = cvCreateMat(size.height, size.width, CV_8UC3);
        cvConvertScale(src, tmp, 255, 127);
        if (doColorConversion)
            cvCvtColor(tmp, tgt, colorConversion);
        else
            cvCopy(tmp, tgt, NULL);
        cvReleaseMat(&tmp);
      }
      break;
    default:
      CV_Assert(0 && "Don't know how to handle that type");
    }
}

void cvConvertArrToIntArray(JavaIntArray bytes, CvMat* img)
{
    // Converts
    CvSize size = cvGetSize(img);
    CvMat dest;
    cvInitMatHeader(&dest, size.height, size.width, CV_8UC4, bytes.ptr);
    cvZScaleTo8U(&dest, img);
}

void cvCopyRectangleToByteArray(JavaByteArray bytes, CvMat* img, size_t offset,
                                CvRect* rectangle)
{
    CvMat tmp;
    cv::Mat m(cvGetMat(img, &tmp));

    unsigned rowlen, firstRow, nRows, firstCol;
    if (rectangle != NULL) {
        rowlen = rectangle->width * m.elemSize();
        firstRow = rectangle->y;
        nRows = rectangle->height;
        firstCol = rectangle->x;
    } else {
        rowlen = m.cols * m.elemSize();
        firstRow = 0;
        nRows = m.rows;
        firstCol = 0;
    }
    unsigned length = rowlen * nRows;

    // Some sanity checks
    CV_Assert(m.step > 0); // can fail for single-row images

    if (offset + length > bytes.size) {
        std::ostringstream msg;
        msg << "Byte array is too small: " << bytes.size << " allocated, "
            << length << " needed.";
        throw cv::Exception(CV_StsBadSize, msg.str(), "cvCopyArrToByteArray",
                __FILE__, __LINE__);
    }

    if (m.isContinuous() && rectangle == NULL)
    {
        // matrix is continuous, we can memcpy the whole area
        memcpy(bytes.ptr + offset, m.data, length);

    } else {

        // If matrix is non-continuous, it is a sequence of rows
        // where a pointer to each row can be obtained with m.ptr,
        // and each row looks like this:
        //
        // <-     -> firstCol columns preceding this rectangle
        // xxxxxxxxx.......................................xxxxxxxxxxxx
        //          <- data length is nCols * elemSize() -><- garbage->
        // <------------- total size is step ------------------------->
        // (we don't increment by "step", we use "m.ptr" instead)
        //
        // Let's memcpy it row-by-row
        for (unsigned int row = 0; row < nRows; row++)
            memcpy(bytes.ptr + offset + row * rowlen,
                   m.ptr(firstRow + row) + firstCol * m.elemSize(),
                   rowlen);
    }
}

void cvCopyArrToByteArray(JavaByteArray bytes, CvMat* img, size_t offset)
{
    cvCopyRectangleToByteArray(bytes, img, offset, NULL);
}

void cvCopyByteArrayToArr(CvMat* img, JavaByteArray data, size_t offset)
{
    CvMat m;
    CvMat* mp;
    mp = cvGetMat(img, &m);
    CV_Assert(mp->step > 0);       // can fail for single-row images
    size_t s = mp->rows * mp->step;
    CvSize is = cvGetSize(img);
    if (data.size <= offset)
    {
        fprintf(stderr, "Offset too large in cvCopyByteArrayToArr: %zd/%zd",
                offset, data.size);
        CV_Assert(0);
    }
    if (data.size - offset > s)
    {
        fprintf(stderr, "Target CvMat too small in cvCopyByteArrayToArr: %zd vs %zd (%d %d, %d %d)",
            data.size, s,
            mp->rows, mp->step,
            is.width, is.height);
        CV_Assert(0);
    }
    memcpy(mp->data.ptr, data.ptr + offset, s);
}


void cvCopyArrToFloatArray(JavaFloatArray floats, CvMat* img, size_t offset)
{
    float * data;
    CvSize size;
    int step;
    cvGetRawData(img, (uchar**)&data, &step, &size);
    CV_Assert(step > 0);           // can fail for single-row images
    unsigned length = size.height * step;
    unsigned floatLength = length / sizeof(float);
    if (offset + floatLength > floats.size)
    {
        fprintf(stderr, "Float array too small in cvCopyArrToFloatArray: %d vs %zd",
                floatLength, floats.size);
        exit(1);
    }
    memcpy(floats.ptr + offset, data, length);
}

void cvCopyArrToDoubleArray(JavaDoubleArray doubles, CvMat* img, size_t offset)
{
    double * data;
    CvSize size;
    int step;
    cvGetRawData(img, (uchar**)&data, &step, &size);
    CV_Assert(step > 0);           // can fail for single-row images
    unsigned length = size.height * step;
    unsigned doubleLength = length / sizeof(double);
    if (offset + doubleLength > doubles.size)
    {
        fprintf(stderr, "Double array too small in cvCopyArrToDoubleArray: %d vs %zd",
                doubleLength, doubles.size);
        exit(1);
    }
    memcpy(doubles.ptr + offset, data, length);
}

void cvZCopy(const CvMat* src, CvMat* dst)
{
    cvCopy(src, dst, NULL);
}

CvMat* cvZQueryFrame(CvCapture* capt)
{
    IplImage* img = cvQueryFrame(capt);
    if (img == NULL)
      return NULL;
    CvMat* res = cvCreateMatHeader(1, 1, CV_8UC1);
    cvGetMat(img, res);
    return res;
}

int cvZWriteFrame(CvVideoWriter* writer, CvMat* mat)
{
    IplImage hdr;
    IplImage* img = cvGetImage(mat, &hdr);
    return cvWriteFrame(writer, img);
}

template <typename T>
void cvZSetX_T(CvMat* mat)
{
    for (int row = 0; row < mat->rows; row++)
    {
        uchar* rowPtrRaw = mat->data.ptr + row * mat->step;
        T* rowPtr = (T*) rowPtrRaw;
        for (int col = 0; col < mat->cols; col++)
          rowPtr[col] = (T) col;
    }
}

// Set each value in mat to be the x coordinate of that matrix cell
void cvZSetX(CvMat* mat)
{
    int elemType = cvGetElemType(mat);
    switch(elemType)
    {
    case CV_8UC1:
      cvZSetX_T<uchar>(mat);
      break;
    case CV_32FC1:
      cvZSetX_T<float>(mat);
      break;
    case CV_64FC1:
      cvZSetX_T<double>(mat);
      break;
    default:
      CV_Assert(false && "Unknown element type for cvZSetX");
    }
}

template <typename T>
void cvZSetY_T(CvMat* mat)
{
    for (int row = 0; row < mat->rows; row++)
    {
        uchar* rowPtrRaw = mat->data.ptr + row * mat->step;
        T* rowPtr = (T*) rowPtrRaw;
        for (int col = 0; col < mat->cols; col++)
          rowPtr[col] = (T) row;
    }
}

// Set each value in mat to be the y coordinate of that matrix cell
void cvZSetY(CvMat* mat)
{
    int elemType = cvGetElemType(mat);
    switch(elemType)
    {
    case CV_8UC1:
      cvZSetY_T<uchar>(mat);
      break;
    case CV_32FC1:
      cvZSetY_T<float>(mat);
      break;
    case CV_64FC1:
      cvZSetY_T<double>(mat);
      break;
    default:
      CV_Assert(false && "Unknown element type for cvZSetY");
    }
}

%}
