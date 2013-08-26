#define CVAPI(a) a
#define CV_INLINE inline
#define CV_CDECL

%module(directors="1") zopencv

// This is recommended in the swig documentation for enums.
// See http://www.swig.org/Doc2.0/Java.html#Java_constants
// for the meaning of javaconst; if a future version of opencv
// has more constant definitions that are not valid java,
// they need to have a %javaconst(0) declaration below.
%javaconst(1);
// CV_RNG_COEFF is an unsigned that doesn't fit in a java int
%javaconst(0) CV_RNG_COEFF;
%include "enums.swg"

%{
// Required for getting the proper version of the Windows API
// We do it first thing here here so OpenCV doesn't define it for us
#ifdef Z_WIN32
#define WIN32_LEAN_AND_MEAN
// windows.h has macros called min and max (which then interfere with
// std::min and std::max), NOMINMAX tells it not to define them.
#define NOMINMAX
#define _WIN32_WINNT 0x0500
#endif

#include <math.h>

namespace {
JNIEnv * JNU_GetEnv();
void callGc();
}  // namespace
%}

%ignore cvGetImage;
%rename(cvIplDepth) cvCvToIplDepth;
%rename(Exception) cvException;


%ignore cvInitNArrayIterator;
%ignore cvMixChannels;
%ignore cvCalcCovarMatrix;
%ignore cvCalcArrHist;
%ignore cvCalcArrBackProject;
%ignore cvCalcArrBackProjectPatch;

// %ignore CvEM;
%ignore CvEM::get_covs;
// %ignore CvEMParams;
%ignore CvEMParams::covs;

// wrapping is wrong, returns new object
%ignore cvInitMatHeader;

/* IplImage* */
%ignore cvQueryFrame;
%ignore cvCreateImageHeader;
%ignore cvInitImageHeader;
%ignore cvCreateImage;
%ignore cvCloneImage;
%ignore cvSetImageCOI;
%ignore cvGetImageCOI;
%ignore cvSetImageROI;
%ignore cvGetImageROI;
%ignore cvResetImageROI;

%ignore cvPyrSegmentation;
%ignore cvSnakeImage;
%ignore cvCheckChessboard;

%ignore cvLoadImage;
%ignore cvDecodeImage;
%ignore cvRetrieveFrame;
%ignore cvWriteFrame;

// static members not linkable with VC
%ignore CvModule;
%ignore CvType;

// Implementation missing in 2.3.1
%ignore cv::MatConstIterator::MatConstIterator;
%ignore cv::getConvertElem;
%ignore cv::getConvertScaleElem;
%ignore cv::SparseMat::ptr;
%ignore cv::SparseMatIterator::SparseMatIterator;
%ignore cv::RTreeClassifier::safeSignatureAlloc;
%ignore cv::OneWayDescriptorBase::ConvertDescriptorsArrayToTree;
%ignore cv::PlanarObjectDetector::getModelROI;
%ignore cv::PlanarObjectDetector::getDetector;
%ignore cv::PlanarObjectDetector::getClassifier;
%ignore cv::Mat::zeros;
%ignore cv::Mat::ones;
// Qt functions -- (empty) implementation missing for Windows
%ignore cvFontQt;
%ignore cvAddText;
%ignore cvDisplayOverlay;
%ignore cvDisplayStatusBar;
%ignore cvCreateOpenGLCallback;
%ignore cvSaveWindowParameters;
%ignore cvLoadWindowParameters;
%ignore cvStartLoop;
%ignore cvStopLoop;
%ignore cvCreateButton;

%include "arrays_java.i";
%include "std_string.i";
%include "various.i";
%include "exception.i"

%exception {
  try {
    $function
  } catch(const cv::Exception &e) {
    SWIG_exception(SWIG_RuntimeError, e.err.c_str());
  }
}

// Use 1.3 convention of private internal methods (we may
// get, say, two constructors with the same arity otherwise,
// confusing non-typehinted clojure).
// Incompatible with the new 'nspace' mechanism, which we don't use.
// See http://www.swig.org/Release/CHANGES 2010-03-06: wsfulton
SWIG_JAVABODY_METHODS(protected, protected, SWIGTYPE)
%pragma(java) jniclassclassmodifiers = "class"

%rename(ttemp1) temp1;
%rename(utemp1) Temp1;
%rename(ttemp1) getTemp1;
%rename(ttemp2) temp2;
%rename(utemp2) Temp2;
%rename(ttemp2) getTemp2;

/* Add some code to the proxy class of the CvArr array type for converting between type used in
* JNI class (long[]) and type used in proxy class ( CvArr[] ) */

%typemap(jni) const CvMat** "jlongArray"
%typemap(jtype) const CvMat** "long[]"
%typemap(jstype) const CvMat** "CvMat[]"
%typemap(javain) const CvMat** "CvMat.cArrayUnwrap($javainput)"

%typemap(in) const CvMat** %{
{
    jlong* arr = jenv->GetLongArrayElements($input, 0);
    int arraylen = jenv->GetArrayLength($input);

    $1 = new CvMat*[arraylen];

    for (int i = 0; i < arraylen; i++) {
        $1[i] = (CvMat*)arr[i];
    }
    jenv->ReleaseLongArrayElements($input, arr, 0);
}
%}

%typemap(argout) const CvMat** %{
{
    delete $1;
}
%}

%typemap(javabody) CvMat %{
  private long swigCPtr;
  protected boolean m_cMemOwn;
  /** Whether it is legal for this object to get garbage
   * collected without first being close()'d.
   */
  protected boolean m_isLightWeight;
  protected boolean m_isRefCounted = true;
  protected Object m_allocData;

  protected CvMat(long cPtr, boolean cMemOwn) {
      m_cMemOwn = cMemOwn;
      swigCPtr = cPtr;
      if (zopencv.cvZOwnsData(this) == 0) {
        setLightWeight(true);
      }
      // Allow collection of allocation data
      m_allocData = com.zenrobotics.zopencv.ZOpenCVUtil.allocated(this);
  }

  public static long getCPtr(CvMat obj) {
    return (obj == null) ? 0 : obj.swigCPtr;
  }

  public void setLightWeight(boolean lightWeight)
  {
    m_isLightWeight = lightWeight;
  }

  public void setRefCounted(boolean refCounted)
  {
    m_isRefCounted = refCounted;
  }

  public boolean isLightWeight()
  {
    return m_isLightWeight;
  }

%}

%typemap(javadestruct, methodname="delete", methodmodifiers="private synchronized") CvMat %{
{
    if (m_cMemOwn)
    {
        boolean wasNull = (swigCPtr == 0);
        freeReference();
        com.zenrobotics.zopencv.ZOpenCVUtil.deallocated(this, m_allocData, wasNull);
    }
}
%}

%typemap(javacode) CvSize %{
    public String toString()
    {
        return String.format("CvSize(%s,%s)",
                    getWidth(),
                    getHeight());
    }
%}

%typemap(javacode) CvPoint %{

  public CvPoint(int x, int y) {
    this(zopencvJNI.new_CvPoint(), true);
    this.setX(x);
    this.setY(y);
  }

  public String toString() {
    return String.format("CvPoint(x=%d y=%d)",
                         this.getX(),
                         this.getY());
  }
%}

%typemap(javainterfaces) CvMat %{ java.io.Closeable %}
%typemap(javacode) CvMat %{

    private int m_refcount = 1;

    public synchronized void incRef() {
        if (! m_isRefCounted) {
            throw new Error("calling incRef on a non-refcounted instance");
        }
        m_refcount++;
    }

    public synchronized void decRef() {
        if (! m_isRefCounted) {
            throw new Error("calling decRef on a non-refcounted instance");
        }
        close();
    }

    // Needs to be synchronized to avoid a race leading to memory
    // corruption. This gets called by dispose and delete.
    protected synchronized void freeReference()
    {
        if (swigCPtr != 0) {
            freeReferenceImpl();
            swigCPtr = 0;
        }
    }

    // backward-compatibility for direct callers of dispose
    public synchronized void dispose() { close(); }
    // Implementation of Closeable
    public synchronized void close()
    {
      m_refcount--;
      if (m_isRefCounted && m_refcount < 0) {
          com.zenrobotics.zopencv.ZOpenCVUtil.doubleDisposed(this, m_allocData);
          throw new Error("double dispose");
      }
      if (m_refcount == 0) {
        freeReference();
      }
    }

    protected static long[] cArrayUnwrap($javaclassname[] arrayWrapper)
    {
        long[] cArray = new long[arrayWrapper.length];
        for (int i=0; i<arrayWrapper.length; i++)
            cArray[i] = $javaclassname.getCPtr(arrayWrapper[i]);
        return cArray;
    }

    public CvMat setSubRect(int x, int y, CvMat subrect)
    {
        zopencv.cvZCopy(subrect,
                        this.getSubRect(x,
                                        y,
                                        subrect.getCols(),
                                        subrect.getRows()));
        return this;
    }

    // TODO: Reimplement on C side without creating new CvRect
    // in Java.
    public CvMat getSubRect(int x, int y, int cols, int rows)
    {
       CvRect rect = new CvRect(x, y, cols, rows);
       return getSubRect(rect);
    }

    public CvMat getSubRect(CvRect rect)
    {
       CvMat header = zopencv.cvCreateMatHeader(1,
                                                1,
                                                zopencv.cvGetElemType(this));
       zopencv.cvGetSubRect(this, header, rect);
       header.setLightWeight(true);
       return header;
    }

    public CvMat toMat()
    {
        return this;
    }

    public CvMat getSubRectCopy(int x, int y, int cols, int rows)
    {
       CvMat arrCopy = zopencv.cvCreateMat(rows,
                                           cols,
                                           zopencv.cvGetElemType(this));
       zopencv.cvZCopy(this.getSubRect(x, y, cols, rows), arrCopy);
       return arrCopy;
    }

    public CvMat reshape(int newChannels, int newRows)
    {
       CvMat header = zopencv.cvCreateMatHeader(1, 1, zopencv.cvGetElemType(this));
       zopencv.cvReshape(this,
               header,
               newChannels,
               newRows);
       header.setLightWeight(true);
       return header;
    }

    public int getRows()
    {
        return zopencv.cvGetSize(this).getHeight();
    }

    public int getCols()
    {
        return zopencv.cvGetSize(this).getWidth();
    }

    public int getElemType()
    {
        return zopencv.cvGetElemType(this);
    }

    private static final java.util.Map<Integer,String> typeStringMap = initTypeStringMap();

    private static java.util.Map<Integer,String> initTypeStringMap()
    {
        String[] ts = {"8U", "8S", "16U", "16S", "32S", "32F",
                       "64F", "8UC1", "8UC2", "8UC3", "8UC4",
                       "8SC1", "8SC2", "8SC3", "8SC4", "16UC1",
                       "16UC2", "16UC3", "16UC4", "16SC1", "16SC2",
                       "16SC3", "16SC4", "32SC1", "32SC2", "32SC3",
                       "32SC4", "32FC1", "32FC2", "32FC3", "32FC4",
                       "64FC1","64FC2", "64FC3", "64FC4"};

        java.util.Map<Integer,String> typeStringMap
                = new java.util.HashMap<Integer,String>(ts.length);
        for (int i = 0; i < ts.length; i++) {
            try {
                typeStringMap.put(
                        zopencvConstants.class.getField("CV_"+ts[i]).getInt(null),
                        ts[i]);
            } catch (IllegalAccessException e) {
                throw new Error("Could not match the CvMat type to" +
                                " any known CvMat type string:", e);
            } catch (NoSuchFieldException e) {
                throw new Error("Could not match the CvMat type to" +
                                " any known CvMat type string:", e);
            }
        }
        return typeStringMap;
    }

    public String getTypeString()
    {
        return typeStringMap.get(this.getElemType());
    }

    public String toString()
    {
        return toString(10);
    }

    public String toString(int maxSize)
    {
        if (swigCPtr == 0) {
             return "CvMat DISPOSED";
        }

        CvSize cvSize = zopencv.cvGetSize(this);
        int imgHeight = cvSize.getHeight();
        int imgWidth = cvSize.getWidth();
        int imgChan = zopencv.cvZGetChannels(this);

        StringBuffer outS = new StringBuffer();
        outS.append("CvMat type: ");
        outS.append(this.getTypeString());
        outS.append(" height: ");
        outS.append(imgHeight);
        outS.append(" width: ");
        outS.append(imgWidth);

        if(imgHeight <= maxSize && imgWidth <= maxSize)
        {
            StringBuffer[] sbChannels = new StringBuffer[4];
            String[] channelId = {"1", "2", "3", "alpha"};

            int n = Math.min(imgChan, sbChannels.length);

            for (int i = 0; i < n; i++)
            {
                sbChannels[i] = new StringBuffer();
                sbChannels[i].append("\n\nChannel ");
                sbChannels[i].append(channelId[i]);
                sbChannels[i].append(":\n");
            }
            int datalen = imgHeight*imgWidth;

            java.text.DecimalFormat df = new java.text.DecimalFormat("#.##");
            for (int i = 0; i < datalen; i++)
            {
                double[] values = zopencv.cvGet1D(this, i).getVal();
                int m = Math.min(values.length, n);
                for (int channel = 0; channel < m; channel++)
                {
                    if ( (i % cvSize.getWidth()) == 0)
                        sbChannels[channel].append('\n');
                    sbChannels[channel].append(df.format(values[channel]));
                    sbChannels[channel].append('\t');
                }
            }

            for (int i = 0; i < n; i++)
            {
                outS.append(sbChannels[i]);
            }
        }
        return outS.toString();
    }
%}

%typemap(javacode) CvScalar %{

  public String toString()
  {
    double[] values = this.getVal();
    StringBuffer sb = new StringBuffer();
    for (int i = 0; i < values.length; i++) {
      sb.append(values[i]);
      sb.append(' ');
    }
    return sb.toString();
  }

  public float getVal(int channel)
  {
    double[] vals = this.getVal();
    return (float)vals[channel];
  }

%}

%typemap(javacode) CvRect %{

  public CvRect(int x, int y, int width, int height) {
    this(zopencvJNI.new_CvRect(), true);
    this.setX(x);
    this.setY(y);
    this.setWidth(width);
    this.setHeight(height);
  }

  public String toString(){
    return String.format("x=%d y=%d width=%d height=%d",
                         this.getX(),
                         this.getY(),
                         this.getWidth(),
                         this.getHeight());
  }

%}

%newobject cvCreateMat;
%newobject cvCreateMatHeader;
%newobject cvCloneArr;
%newobject cvCloneMat;
%newobject cvMatFromData;
%newobject cvMatFromDataAndOffset;
%newobject cvLoadImageM;
%newobject cvZLoad;
%newobject cvZQueryFrame;

%newobject cvZCreateVideoWriter;
%newobject cvCreateFileCapture;

%newobject cvCreateStructuringElementEx;

%newobject cvZCreateHist;

%newobject cvCreateStereoBMState;

%newobject cvCreateMemStorage;


%{
#define CvArr IgnoreCvArr
#include "core/types_c.h"
#undef CvArr
%}

%include "typemaps.i";

%{
typedef void CvArr;
#include "cxcore.h"
#include "cv.h"
#include "highgui.h"
#include "zopencv.h"
#include "ml.h"
#include "cxcore.hpp"
%}

%define DISPOSABLE(type, release)
%nodefaultctor type;
%nodefaultdtor type;
%{
void release ## Pointer(type *c) {
    if (!c) return;
    type *c2 = c;
    release (&c2);
}
%}
%typemap(javainterfaces) type %{ java.io.Closeable %}
%typemap(javacode) type %{ public synchronized void close() { delete(); } %}
%typemap(javadestruct, methodname="delete", methodmodifiers="public synchronized") type %{
{
    if(swigCPtr != 0 && swigCMemOwn) {
      swigCMemOwn = false;
      zopencvJNI. ## release ## Pointer(swigCPtr, this);
    }
    swigCPtr = 0;
}
%}
%javamethodmodifiers release ## Pointer( type *c) "protected";
%ignore release;
void release ##Pointer(type *c);
%enddef
DISPOSABLE(CvCapture, cvReleaseCapture)
DISPOSABLE(CvVideoWriter, cvReleaseVideoWriter)
DISPOSABLE(CvStereoBMState, cvReleaseStereoBMState)
DISPOSABLE(CvHistogram, cvReleaseHist)
DISPOSABLE(IplConvKernel, cvReleaseStructuringElement)
DISPOSABLE(CvMemStorage, cvReleaseMemStorage)

%{
#include <cstdio>

#include "shiftedops.h"


int cvZOwnsData(CvMat* img)
{
  // CvMats have a field called refcount that OpenCV folks
  // had at some point planned to use for reference counting,
  // but it didn't work out. It's now used just as an indicator
  // that the CvMat owns the data: if it's non-null the CvMat
  // does own the data.
  return img->refcount == 0 ? 0 : 1;
}

/* Ugly kludge fix */
void cvGetState(CvArr* arr, int out[4])
{
    IplImage tmpImg;
    IplImage* img = cvGetImage(arr, &tmpImg);
    CvSize s = cvGetSize(img);
    out[0] = cvGetElemType(img);
    out[1] = img->nChannels;
    out[2] = s.width;
    out[3] = s.height;
}

CvVideoWriter* cvZCreateVideoWriter(const char* filename,
    const char* fourcc, double fps, CvSize frame_size,
    int is_color) {
  return cvCreateVideoWriter(filename,
      CV_FOURCC(fourcc[0], fourcc[1], fourcc[2], fourcc[3]),
      fps, frame_size, is_color);
}


CvRNG cvZRNG(long seed)
{
    return cvRNG(seed);
}

void cvZReleaseFileStorage(CvFileStorage* fs)
{
    cvReleaseFileStorage(&fs);
}



%}

#define CvArr IgnoreCvArr
%include "core/types_c.h"
struct JavaFloatArray;

%ignore cvReleaseMat;
#undef CvArr

%extend CvMat
{
    void printPtr()
    {
        printf("CvMat ptr: %p\n", $self);
    }
    CvMat* toArr()
    {
        return $self;
    }
    void freeReferenceImpl()
    {
        CvMat* p = $self;
        // printf("Release mat %p\n", p);
        cvReleaseMat(&p);
    }
};

%extend IplImage
{
    void printPtr()
    {
        printf("IplImage ptr: %p (imagedata %p w %d h %d nChannels %d)\n", $self,
            $self->imageData,
            $self->width,
            $self->height,
            $self->nChannels);
    }
    void freeReferenceImpl()
    {
        IplImage* p = $self;
        // printf("Release image %p\n", p);
        cvReleaseImage(&p);
    }
};

/* For SWIG only, define CvArr to CvMat -> only get CvMat
 * versions of all funcs.
 */
#define CvArr CvMat
%include "cxcore.h"
%include "cv.h"
%ignore noArray;
%ignore AdjusterAdapter;
%include "core/core.hpp"
%include "core/core_c.h"
%include "imgproc/types_c.h"
%include "imgproc/imgproc_c.h"
/* CvCapture and CvVideoWriter are declared as opaque types in
   opencv, we want to say they are known structs to get java
   classes for them (rather than just the pointer classes). */
struct CvCapture {};
struct CvVideoWriter {};
%include "highgui.h"
%include "highgui/highgui_c.h"
%include "calib3d/calib3d.hpp"
%include "features2d/features2d.hpp"
%ignore HOGDescriptor;
%include "objdetect/objdetect.hpp"
%include "ml/ml.hpp"
%include "zopencv.h"
%include "ml.h"
#undef CvArr

%ignore prepShifted;
%ignore finishShifted;
%ignore zadd;
%include "shiftedops.h"

%{

/* Return the number of channels in the image */
int cvZGetChannels(CvMat *m)
{
  int elemType = cvGetElemType(m);
  return CV_MAT_CN(elemType);
}

%}


CvVideoWriter* cvZCreateVideoWriter(const char* filename,
    const char* fourcc, double fps, CvSize frame_size,
    int is_color);

//jobjectArray cvExtractSIFTFastSimple(CvMat* img);

int cvZOwnsData(CvMat* img);

#undef cvConvert
void cvConvert(CvMat* src, CvMat* dst);

void cvGetState(CvMat* arr, int out[4]);


CvRNG cvZRNG(long seed);

void cvZReleaseFileStorage(CvFileStorage* fs);


int cvZGetChannels(CvMat *);

%feature("director") ZMouseCallback;
class ZMouseCallback {
 public:
  virtual ~ZMouseCallback();
  virtual void callback(int event, int x, int y, int flags);
};



%{
class ZMouseCallback {
 public:
  virtual ~ZMouseCallback();
  virtual void callback(int event, int x, int y, int flags);
};

namespace {
void ZMCcallback(int event, int x, int y, int flags, void* param) {
  ZMouseCallback* cb = (ZMouseCallback*)param;
  cb->callback(event, x, y, flags);
}
}  // namespace

void cvZSetMouseCallback(const char* window_name, ZMouseCallback* cb) {
  cvSetMouseCallback(window_name, &ZMCcallback, cb);
}

ZMouseCallback::~ZMouseCallback() { }
void ZMouseCallback::callback(int event, int x, int y, int flags) { }

%}

%pragma(java) jniclasscode=%{
    static { if (System.getenv("ZNO_OPENCV") != null) {
               throw new Error("loading opencv disallowed");
             }
             System.loadLibrary("zopencv");
           }
    // The native allocator prints stacktrace on failure, make
    // it convenient to do via JNI.
    public static void printStackTrace()
    {
        new java.lang.Exception().printStackTrace();
        System.err.flush();
    }
%}

// Custom memory allocation for OpenCV: since we wrap the returned memory
// in GC'd Java objects we need to also trigger GC if we run over our
// desired limit or can't get any more.
//

%include "zalloc.i"

%{

#ifdef Z_WIN32
#include <windows.h>
#else
#include <pthread.h>
#endif

#include <jni.h>
#ifdef __linux__
#endif

namespace {

JavaVM *cached_jvm = 0;

JNIEnv * JNU_GetEnv()
{
    JNIEnv *env = NULL;
    jint rc = cached_jvm->GetEnv((void **)&env, JNI_VERSION_1_2);
    if (rc == JNI_EDETACHED) {
        jint attachSuccess = cached_jvm->AttachCurrentThread((void**)&env,NULL);
        if (attachSuccess != 0) {
            printf("Failed to attach JNI thread %d\n", (int) attachSuccess);
            return NULL;
        }
    }
    if (rc == JNI_EVERSION) {
        CV_Assert(false);
        return NULL;
    }
    CV_Assert(env);
    return env;
}

void printStackTrace()
{
    JNIEnv *env = JNU_GetEnv();
    if (!env) return;
    jclass zopencvJNI = env->FindClass("com/zenrobotics/zopencv/impl/zopencvJNI");
    CV_Assert(zopencvJNI != NULL);
    jmethodID printStackTrace = env->GetStaticMethodID(zopencvJNI, "printStackTrace", "()V");
    CV_Assert(printStackTrace);
    env->CallVoidMethod(zopencvJNI, printStackTrace, 0);
 }

void callGc()
{
    // We could cache the class and method ids but we assume that this code
    // path should not be hit regularly in production (we want to use the
    // incremental GC and have it run regularly).
    printf("Oom -> try Java GC\n");
    JNIEnv *env = JNU_GetEnv();
    if (!env) return;
    jclass system = env->FindClass("java/lang/System");
    CV_Assert(system != NULL);
    jmethodID gc = env->GetStaticMethodID(system, "gc", "()V");
    jmethodID runFinalization = env->GetStaticMethodID(system, "runFinalization", "()V");
    CV_Assert(gc);
    CV_Assert(runFinalization);
    // We want to free opencv memory wrapped in Java objects that free the
    // memory in their finalizers. To do that we need to both find the
    // wrappers that are eligible for releasing (the first GC call) and call
    // their finalizers. This may get called several times if a single call
    // doesn't release memory.
    printf("Oom ---> gc\n");
    env->CallVoidMethod(system, gc, 0);
    printf("Oom ---> finalization\n");
    env->CallVoidMethod(system, runFinalization, 0);
    printf("Oom Java GC done\n");
}

// Contrary to the SWIG documentation using java byte arrays for allocations
// doesn't help as any pinned memory (either via JNI GetByteArrayElements or
// java.nio) will be allocated outside the java heap (as the JVM wants it
// all to be movable) and will not cause memory pressure in the java heap.
//
// We need two versions of locking as -mno-cygwin doesn't have pthreads.
struct JavaOomData : public OomData {
    JavaOomData()
    {
#ifdef Z_WIN32
        // 0x80000400 comes from Microsoft's example (MSDN doc),
        // the high bit means that the resources are allocated here
        // rather than on the first call EnterCriticalSection (which
        // is thus guaranteed to succeed). 400 is the spin count.
        CV_Assert(InitializeCriticalSectionAndSpinCount(&m_mutex,
            0x80000400));
#else
        pthread_mutex_init(&m_mutex, NULL);
#endif
        if (sizeof(void*) > 4) {
            m_limit = 3L * 1024 * 1024 * 1024;
        } else {
            m_limit = 1024 * 1024 * 1024;
        }
        m_print_stacks = getenv("ZOPENCV_ALLOC_DEBUG");

    }
    virtual void OomCallback() { callGc(); }
    virtual void* Allocate(size_t s) {
        void *ret = 0;
        ret = malloc(s);
        if (!ret)
            printf("JavaOomAlloc: returning null!\n");
        if (m_print_stacks) {
            Lock();
            fflush(stderr);
            fprintf(stderr, "ZOPENCV allocate stack for %p size %lu\n", ret, (unsigned long)s);
            fflush(stderr);
            printStackTrace();
            Unlock();
        }
        return ret;
    }
    virtual void Deallocate(void* p) {
        if (m_print_stacks) {
            Lock();
            fflush(stderr);
            fprintf(stderr, "ZOPENCV free stack for %p\n", p);
            fflush(stderr);
            printStackTrace();
            Unlock();
        }
        free(p);
    }
    virtual void Lock()
    {
#ifdef Z_WIN32
        EnterCriticalSection(&m_mutex);
#else
        pthread_mutex_lock(&m_mutex);
#endif
    }
    virtual void Unlock()
    {
#ifdef Z_WIN32
        LeaveCriticalSection(&m_mutex);
#else
        pthread_mutex_unlock(&m_mutex);
#endif
    }
#ifdef Z_WIN32
    CRITICAL_SECTION m_mutex;
#else
    pthread_mutex_t m_mutex;
#endif
    bool m_print_stacks;
    bool m_use_duma;
};

int NopErrorHandler(int /* status */, const char* /* func_name */,
                    const char* /* err_msg */, const char* /* file_name */,
                    int /* line */, void*) {
  // cv::error will throw an exception, we don't want to do anything
  // where it's thrown
  return 0;
}

}  // namespace

static JavaOomData* s_oomData;

JNIEXPORT jint JNICALL JNI_OnLoad(JavaVM *jvm, void *reserved)
{
#ifdef Z_WIN32
  // Processes spawned by cygwin have disabled the just-in-time
  // debugger/WER dialog - re-enable here.
  ::SetErrorMode(0);
#endif
  cached_jvm = jvm;
  s_oomData = new JavaOomData;
  cvSetMemoryManager(&cvOomAlloc, &cvOomFree, s_oomData);
  cvRedirectError(&NopErrorHandler);
  return JNI_VERSION_1_2;
}

JNIEXPORT void JNICALL JNI_OnUnload(JavaVM *jvm, void *reserved)
{
    cached_jvm = 0;
    cvSetMemoryManager(NULL, NULL, NULL);
    delete s_oomData;
    s_oomData = 0;
    return;
}

OomData* getOomData()
{
    return s_oomData;
}

%}

// Declare the members we want do expose
struct OomData
{
private:
    OomData();
public:
    int m_size;
    size_t m_limit;
    int m_limitCounter;
    int m_limitMax;
    int m_nAllocs;
    int m_nFrees;
    int m_nOOMs;
    int m_nHardOOMs;

};

OomData* getOomData();

%include "sick.i"
%include "unpacku12.i"
%include "ml.i"
%include "arith.i"
%include "data.i"
%include "segmentation.i"
%include "misc.i"
