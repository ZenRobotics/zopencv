%typemap(in) JavaFloatArray "$1.ptr = jenv->GetFloatArrayElements($input, 0); $1.size = jenv->GetArrayLength($input);"
%typemap(argout) JavaFloatArray "jenv->ReleaseFloatArrayElements($input, $1.ptr, 0);"
%typemap(jtype) JavaFloatArray "float[]"
%typemap(jstype) JavaFloatArray "float[]"
%typemap(jni) JavaFloatArray "jfloatArray"
%typemap(javain) JavaFloatArray "$javainput"

%typemap(in) JavaByteArray "$1.ptr = jenv->GetByteArrayElements($input, 0); $1.size = jenv->GetArrayLength($input);"
%typemap(argout) JavaByteArray "jenv->ReleaseByteArrayElements($input, $1.ptr, 0);"
%typemap(jtype) JavaByteArray "byte[]"
%typemap(jstype) JavaByteArray "byte[]"
%typemap(jni) JavaByteArray "jbyteArray"
%typemap(javain) JavaByteArray "$javainput"

%typemap(in) JavaIntArray "$1.ptr = jenv->GetIntArrayElements($input, 0); $1.size = jenv->GetArrayLength($input);"
%typemap(argout) JavaIntArray "jenv->ReleaseIntArrayElements($input, $1.ptr, 0);"
%typemap(jtype) JavaIntArray "int[]"
%typemap(jstype) JavaIntArray "int[]"
%typemap(jni) JavaIntArray "jintArray"
%typemap(javain) JavaIntArray "$javainput"

%typemap(in) JavaDoubleArray "$1.ptr = jenv->GetDoubleArrayElements($input, 0); $1.size = jenv->GetArrayLength($input);"
%typemap(argout) JavaDoubleArray "jenv->ReleaseDoubleArrayElements($input, $1.ptr, 0);"
%typemap(jtype) JavaDoubleArray "double[]"
%typemap(jstype) JavaDoubleArray "double[]"
%typemap(jni) JavaDoubleArray "jdoubleArray"
%typemap(javain) JavaDoubleArray "$javainput"

%typemap(out) JavaDirectByteBuffer { $result = jenv->NewDirectByteBuffer($1.ptr, $1.size); }
%typemap(javaout) JavaDirectByteBuffer { return $jnicall; }
%typemap(jtype) JavaDirectByteBuffer "java.nio.ByteBuffer"
%typemap(jstype) JavaDirectByteBuffer "java.nio.ByteBuffer"
%typemap(jni) JavaDirectByteBuffer "jobject"

%typemap(in) CvPoint2D32fArrayAsFloats "
$1.ptr = jenv->GetFloatArrayElements($input, 0);
$1.size = jenv->GetArrayLength($input);
$1.nPoints = $1.size / 2;
$1.pointPtr = new CvPoint2D32f[$1.nPoints];
for (int i = 0; i < $1.nPoints; i++)
{
    $1.pointPtr[i].x = $1.ptr[2 * i + 0];
    $1.pointPtr[i].y = $1.ptr[2 * i + 1];
}
"
%typemap(argout) CvPoint2D32fArrayAsFloats "
for (int i = 0; i < $1.nPoints; i++)
{
    $1.ptr[2 * i + 0] = $1.pointPtr[i].x;
    $1.ptr[2 * i + 1] = $1.pointPtr[i].y;
}

jenv->ReleaseFloatArrayElements($input, $1.ptr, 0); delete[] $1.pointPtr;
"
%typemap(jtype) CvPoint2D32fArrayAsFloats "float[]"
%typemap(jstype) CvPoint2D32fArrayAsFloats "float[]"
%typemap(jni) CvPoint2D32fArrayAsFloats "jfloatArray"
%typemap(javain) CvPoint2D32fArrayAsFloats "$javainput"

%typemap(in) CvPointArrayAsInts "
$1.ptr = jenv->GetIntArrayElements($input, 0);
$1.size = jenv->GetArrayLength($input);
$1.nPoints = $1.size / 2;
$1.pointPtr = new CvPoint[$1.nPoints];
for (int i = 0; i < $1.nPoints; i++)
{
    $1.pointPtr[i].x = $1.ptr[2 * i + 0];
    $1.pointPtr[i].y = $1.ptr[2 * i + 1];
}
"
%typemap(argout) CvPointArrayAsInts "
for (int i = 0; i < $1.nPoints; i++)
{
    $1.ptr[2 * i + 0] = $1.pointPtr[i].x;
    $1.ptr[2 * i + 1] = $1.pointPtr[i].y;
}

jenv->ReleaseIntArrayElements($input, $1.ptr, 0); delete[] $1.pointPtr;
"
%typemap(jtype) CvPointArrayAsInts "int[]"
%typemap(jstype) CvPointArrayAsInts "int[]"
%typemap(jni) CvPointArrayAsInts "jintArray"
%typemap(javain) CvPointArrayAsInts "$javainput"

%{
struct JavaFloatArray
{
    size_t size;
    jfloat* ptr;
};

struct JavaByteArray
{
    size_t size;
    jbyte* ptr;
};

struct JavaIntArray
{
    size_t size;
    jint* ptr;
};

struct JavaDoubleArray
{
    size_t size;
    jdouble* ptr;
};

struct CvPoint2D32fArrayAsFloats : JavaFloatArray
{
    int nPoints;
    CvPoint2D32f* pointPtr;
};

struct CvPointArrayAsInts : JavaIntArray
{
    int nPoints;
    CvPoint* pointPtr;
};

struct JavaDirectByteBuffer
{
    size_t size;
    void* ptr;
};
%}
