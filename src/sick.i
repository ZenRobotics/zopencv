
%newobject cvZFind;
CvMat* cvZFind (CvMat* _image, CvMat* _mask);

%newobject cvZHeight;
CvMat* cvZHeightfieldVolumeCov (CvMat* _height, CvMat* _mask);

%{

// Covariance matrix of masked volume with height > 0.
CvMat* cvZHeightfieldVolumeCov (CvMat* _height, CvMat* _mask)
{
  using namespace cv;
  Mat_<float> height (_height);
  Mat_<unsigned char> mask (_mask);

  Matx31d mx (0,0,0);
  double sum_z = 0;

  // Weighted sum of squares
  // i.e. E XX^T.
  Matx33d EXXT (0, 0, 0,
                0, 0, 0,
                0, 0, 0);

  for (int y = 0; y < height.rows; ++y)
    for (int x = 0; x < height.cols; ++x)
      if (mask.at<unsigned char>(y,x))
        {
          double z = height.at<float>(y,x);
          if (z > 0)
            {
              // Mean of distribution
              Matx31d cx (x, y, 0.5*z);
              mx    += z * cx;
              sum_z += z;
              // Covariance of  uniform distribution at pixel
              Matx33d S(1.0/12.0,        0,          0,
                               0, 1.0/12.0,          0,
                               0,        0, 1/12.0*z*z);
              Matx33d xxt = cx * cx.t();
              EXXT  += z * (S + xxt);
            }
        }

  // Covariance from weighted sum of squares and mean.
  mx = (1.0/sum_z) * mx;

  Matx33d C = (1.0 / sum_z) * EXXT - mx * mx.t();
  CvMat* _C = cvCreateMat (3, 3, CV_64FC1);
  Mat_<double> __C(_C);
  Mat(C).copyTo(__C);
  return _C;
}

// Make matrix of (x, y, image(x,y)) where mask(x,y) > 0.
CvMat* cvZFind (CvMat* _image, CvMat* _mask)
{
  using namespace cv;
  Mat_<float> image (_image);
  Mat_<unsigned char> mask (_mask);
  unsigned n = cvCountNonZero (_mask);
  unsigned i = 0;
  CvMat* _result = cvCreateMat (n, 3, CV_32FC1);
  Mat_<float> result (_result);
  for (int y = 0; y < image.rows; ++y)
    for (int x = 0; x < image.cols; ++x)
      if (mask (y,x) > 0)
        {
          result (i, 0) = x;
          result (i, 1) = y;
          result (i, 2) = image (y,x);
          i++;
        }
  return _result;
};

%}
