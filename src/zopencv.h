
enum { N_HAAR_RECORD = 5,
    HAAR_RECORD_X = 0,
    HAAR_RECORD_Y = 1,
    HAAR_RECORD_W = 2,
    HAAR_RECORD_H = 3,
    HAAR_RECORD_NEIGHBORS = 4,
};

enum { N_SURF_RECORD = 6,
    SURF_RECORD_X = 0,
    SURF_RECORD_Y = 1,
    SURF_RECORD_LAPLACIAN = 2,
    SURF_RECORD_SIZE = 3,
    SURF_RECORD_DIR = 4,
    SURF_RECORD_HESSIAN = 5
};

enum { N_SIFT_RECORD = 4,
    SIFT_RECORD_X = 0,
    SIFT_RECORD_Y = 1,
    SIFT_RECORD_SCALE = 2,
    SIFT_RECORD_ORI = 3,
    N_SIFT_DESCRIP = 128
};
