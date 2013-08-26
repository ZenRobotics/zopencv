package com.zenrobotics.zopencv;
import com.zenrobotics.zopencv.impl.*;

/** Useful static functions related to the OpenCV wrapper.
 */
public final class ZOpenCVUtil implements com.zenrobotics.zopencv.impl.zopencvConstants
{
    /** If set, deallocations of heavyweight objects get warnings.
     */
    private static boolean s_warnOnDealloc;
    /** Number of heavy weight deallocs since startup.
     */
    private static int s_heavyWeightDeallocs;

    private ZOpenCVUtil() { }

    /** If set, warnings are printed on heavyweight object
     * gc. The heavyweight gcs are measured anyway, but this
     * flag causes all CvMat allocations to allocate a Throwable
     * in case they get deallocated badly. Should only be used
     * when debugging.
     */
    public static void setWarnOnDealloc(boolean v)
    {
        s_warnOnDealloc = v;
    }

    public static int getHeavyWeightDeallocs()
    {
        return s_heavyWeightDeallocs;
    }

    public static int iplDepthToCvDepth(int iplDepth)
    {
        // Can't use switch because constants are not constant enough :(
        if (iplDepth == IPL_DEPTH_8U) return CV_8U;
        if (iplDepth == IPL_DEPTH_8S) return CV_8S;
        if (iplDepth == IPL_DEPTH_16U) return CV_16U;
        if (iplDepth == IPL_DEPTH_16S) return CV_16S;
        if (iplDepth == IPL_DEPTH_32S) return CV_32S;
        if (iplDepth == IPL_DEPTH_32F) return CV_32F;
        if (iplDepth == IPL_DEPTH_64F) return CV_64F;
        throw new Error(String.format("Invalid IPL depth given: %d", iplDepth));
    }
    /** Given ipl depth and number of channels, return CvMat
     * elem type.
     */
    public static int iplDepthAndChannelsToMatType(int iplDepth,
            int channels)
    {
        int cvDepth = iplDepthToCvDepth(iplDepth);
        return (CV_MAT_DEPTH_MASK & cvDepth) |
            ((channels - 1) << CV_CN_SHIFT);
    }

    /** PRIVATE, TO BE CALLED BY CvMat.
     */
    public static Object allocated(CvMat m)
    {
        if (s_warnOnDealloc)
            return new Throwable();
        else
            return null;
    }
    public static void deallocated(CvMat m, Object allocData,
            boolean wasNull)
    {
        if (!m.isLightWeight() && !wasNull)
        {
            s_heavyWeightDeallocs++;
            if (s_warnOnDealloc)
            {
                Throwable init = (Throwable)allocData;
                System.err.printf(
                        "Garbage collecting heavyweight CvMat %s\n",
                        m.toString());
                init.printStackTrace(System.err);
            }
        }
    }
    public static void doubleDisposed(CvMat m, Object allocData)
    {
        if (allocData != null) {
            Throwable init = (Throwable)allocData;
            System.err.printf(
                        "double dispose of %s",
                        m.toString());
            init.printStackTrace(System.err);
        }
    }
}
