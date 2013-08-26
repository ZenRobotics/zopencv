// Custom OpenCV allocator that allows calling a custom function
// on out-of-memory and for the raw memory.
//
// Derived from the OpenCV library, licensed under:
//
//                        Intel License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2000, Intel Corporation, all rights reserved.
// Third party copyrights are property of their respective owners.
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
//
//   * Redistribution's of source code must retain the above copyright notice,
//     this list of conditions and the following disclaimer.
//
//   * Redistribution's in binary form must reproduce the above copyright notice,
//     this list of conditions and the following disclaimer in the documentation
//     and/or other materials provided with the distribution.
//
//   * The name of Intel Corporation may not be used to endorse or promote products
//     derived from this software without specific prior written permission.
//
// This software is provided by the copyright holders and contributors "as is" and
// any express or implied warranties, including, but not limited to, the implied
// warranties of merchantability and fitness for a particular purpose are disclaimed.
// In no event shall the Intel Corporation or contributors be liable for any direct,
// indirect, incidental, special, exemplary, or consequential damages
// (including, but not limited to, procurement of substitute goods or services;
// loss of use, data, or profits; or business interruption) however caused
// and on any theory of liability, whether in contract, strict liability,
// or tort (including negligence or otherwise) arising in any way out of
// the use of this software, even if advised of the possibility of such damage.
//

%{

// Pass a pointer to a OomData as userdata to the allocator
// to get a function called on out-of-memory. (Function pointers
// cannot be passed in void*).

#include <map>
using std::map;
#include <set>
#include <stdlib.h>

struct OomData {
    OomData()
    {
        m_size = 0; m_limit = 256 * 1024 * 1024;
        m_limitCounter = 0; m_limitMax = 1;
        m_nAllocs = 0; m_nFrees = 0; m_freed = NULL;

        char *zalloc_debug = getenv("ZALLOC_DEBUG");
        if (zalloc_debug && strlen(zalloc_debug) > 0) {
            fprintf(stderr, "ZALLOC_DEBUG enabled\n");
            m_freed = new std::set<uintptr_t>();
        }
    }
    virtual ~OomData()
    {
        if (m_freed) {
            delete m_freed;
        }
    }
    virtual void OomCallback(void) = 0;
    virtual void* Allocate(size_t) = 0;
    virtual void Deallocate(void*) = 0;
    // The lock is used for manipulating the member variables,
    // not for calling Allocate/Deallocate.
    virtual void Lock(void) = 0;
    virtual void Unlock(void) = 0;
    map<uintptr_t, size_t> m_allocated;
    std::set<uintptr_t>* m_freed;
    size_t m_size;
    size_t m_limit;
    /** Number of times the limit has been hit
     * since the previous OomCallback call
     */
    int m_limitCounter;
    /** The number of times to wait next time before
     * the OomCallback call arising from hitting the
     * limit.
     */
    int m_limitMax;

    /** The number of allocations so far.
     */
    int m_nAllocs;
    /** The number of frees so far.
     */
    int m_nFrees;
    /** The number of OOMs limit hits so far
     */
    int m_nOOMs;
    /** The number of hard OOMs hit so far
     */
    int m_nHardOOMs;
};

// From cxmisc.h
static void* cvOomAlignPtr(const void* ptr, int align = 32)
{
    assert( (align & (align-1)) == 0 );
    return (void*)( ((size_t)ptr + align - 1) & ~(size_t)(align-1) );
}

// Custom allocator for OpenCV. The return value from allocate
// (and hence the input to deallocate) must be compatible
// with the default OpenCV allocator as it may have been called before we
// switch them.

// From icvDefaultAlloc in OpenCV cxalloc.cpp. Returns aligned memory for
// efficiency, puts the pointer to the whole block at return_value - 1.
static void* cvOomAlloc(size_t requested_size, void* userdata)
{
    const size_t OOM_CV_MALLOC_ALIGN = 32;
    const size_t PTR_SIZE = sizeof(char *);
    const int MAX_TRIES = 3;
    OomData* oomD = (OomData*)userdata;
    assert(oomD);
    size_t size =  (size_t)(requested_size +
                            OOM_CV_MALLOC_ALIGN * ((requested_size >= 4096) + 1)
                            + PTR_SIZE);
    int tries = 0;
    bool hitLimit = oomD->m_size + size > oomD->m_limit;
    /* If we hit the limit, see if we should do callback yet
     */
    if (hitLimit)
    {
        oomD->m_nOOMs++;
        if (oomD->m_limitCounter == 0)
        {
            oomD->m_limitMax *= 10;
            oomD->m_limitCounter = oomD->m_limitMax;

            while (hitLimit && tries < MAX_TRIES) {
                printf("OOM Alloc: over limit -> try %d at freeing memory\n",
                    tries);
                oomD->OomCallback();
                ++tries;
            }
        } else
        {
            oomD->m_limitCounter--;
        }
    } else
    {
        oomD->m_limitCounter = 0;
        /* oomD->m_limitMax = 1; */
    }
    char* ptr0 = (char*)oomD->Allocate(size);
    while(!ptr0 && tries < MAX_TRIES) {
      printf("OOM Alloc: null -> try %d at freeing memory\n",
              tries);
        oomD->OomCallback();
        ptr0 = (char*)oomD->Allocate(size);
        ++tries;
    }
    if (!ptr0) {
        printf("OOM Alloc: returning null\n");
        oomD->m_nHardOOMs++;
        return NULL;
    }
    char* ptr =  (char*)cvOomAlignPtr(ptr0 + PTR_SIZE + 1, OOM_CV_MALLOC_ALIGN);
    *(char**)(ptr - sizeof(char*)) = ptr0;
    oomD->Lock();
    oomD->m_allocated[(uintptr_t)ptr] = size;
    if (oomD->m_freed) {
         oomD->m_freed->erase((uintptr_t)ptr);
    }
    oomD->m_size += size;
    oomD->Unlock();
    oomD->m_nAllocs++;
    return ptr;
}

extern "C" int icvDefaultFree(void *ptr, void *userdata);
static int cvOomFree(void* pptr, void* userdata)
{
    if (!pptr) return 0;
    bool we_allocated = false;
    OomData* oomD = (OomData*)userdata;
    assert(oomD);
    oomD->Lock();
    if (oomD->m_freed) {
        std::set<uintptr_t>::iterator i = oomD->m_freed->find((uintptr_t)pptr);
        if (i != oomD->m_freed->end()) {
            fprintf(stderr, "DOUBLE FREE %p\n", pptr);
            abort();
        } else {
            oomD->m_freed->insert((uintptr_t)pptr);
        }
    }
    // TODO: How bad, exactly, does this get, performance-wise?
    // After loading a large classifier tree, this might
    // actually be pretty expensive
    map<uintptr_t, size_t>::iterator i = oomD->m_allocated.find((uintptr_t)pptr);
    if (i != oomD->m_allocated.end()) {
        oomD->m_size -= i->second;
        we_allocated = true;
        oomD->m_allocated.erase(i);
    }
    oomD->Unlock();
    char* real_pointer = *((char**)pptr - 1);
    if (we_allocated) {
        oomD->Deallocate(real_pointer);
    } else {
      // XXX(lrasinen) Edit/remove the ifdef once all OpenCV libraries
      // have been recompiled to export icvDefaultFree
      // Currently the only such platforms are win-64 and win-32
#if defined(_WIN64) || defined(WIN32)
        icvDefaultFree(pptr, NULL);
#else
        free(real_pointer);
#endif
    }
    oomD->m_nFrees++;
    return 0;
}

%}
