(ns example
  (:use com.zenrobotics.zopencv))

(def +img+ (cvLoadImageM "apples.jpg" 1))

(def colors [[1.0 0 0]
             [0 1.0 0]
             [0 0 1.0]

             [0 1.0 1.0]
             [1.0 0 1.0]
             [1.0 1.0 0]

             [0 0.5 1.0]
             [0 1.0 0.5]
             [0.5 0 1.0]
             [1.0 0 0.5]
             [0.5 1.0 0]
             [1.0 0.5 0]])

(defn color [i scale]
  (apply cvScalar (map (partial * scale)
                       (get colors (mod i (count colors))))))

(def +darken-level+ 140)

(defn colorify-components
  [in]
  (let [gray (cvZCreateLike in CV_8UC1)
        ids (cvZCreateLike in CV_16UC1)
        mask (cvZCreateLike in CV_8UC1)
        output (cvZCreateLike in CV_8UC3)]

    ;; Prepare image for segmentation: convert to grayscale,
    ;; threshold, and erode.
    (cvCvtColor in gray CV_RGB2GRAY)
    (cvThreshold gray gray 70 255 CV_THRESH_BINARY)

    (cvErode gray gray nil 2)
    (cvErode gray gray nil 2)

    ;; Prepare output image: darkened version of input
    (cvConvertScale in output (/ +darken-level+ 255.0) 0.0)

    ;; Segment
    (let [nids (cvZColorConnectedComponents2 gray ids)]
      (println "Found" nids "components.")

      ;; Overlay each segment on top of output image using a color.
      (doseq [i (range 1 nids)]
        (let [value (* i 10.0)]
          (cvCmpS ids (double i) mask CV_CMP_EQ)
          (cvAddS output (color i (- 255 +darken-level+))
                  output mask))))
    output))

;; Run this:
;;(cvZSaveImage "/tmp/foo.png" (colorify-components +img+))