(ns com.zenrobotics.zopencv
  (:require
   [clojure.string :as str])
  (:import
   (java.lang.reflect Field Method Modifier)
   com.zenrobotics.zopencv.impl.zopencv))

;; exposing jni functions

(defn java-methods [^Class class]
  (reduce (fn [res ^java.lang.reflect.Method method]
            (merge-with concat res {(.getName method) [method]}))
          {}
          (.getMethods class)))

(def zopencv-methods
  (let [all-methods (java-methods zopencv)
        static-methods (into {}
                             (filter (fn [[k [v]]]
                                       (Modifier/isStatic
                                        (.getModifiers v)))
                                     all-methods))
        cv-methods (into {}
                         (filter (fn [[k [v]]]
                                   (.startsWith (name k) "cv"))
                                 static-methods))]
    cv-methods))

(defn ^Method get-cv [name] (zopencv-methods name))

(defn- make-a-body [^Method method]
  (let [method-name (symbol (.getName method))
        types (.getParameterTypes method)
        arity (count types)
        typed-args (map #(gensym (str "m" % "_")) (range arity))
        typehinted-args (map (fn [s t] (with-meta s {:tag t}))
                             typed-args types)
        rettype (.getReturnType method)]
    (list (vec typed-args)
          (concat ['. 'com.zenrobotics.zopencv.impl.zopencv
                   method-name] typed-args))))

(defmacro defcv [method-name]
  (let [methods (get-cv (str method-name))
        ^Method method (first methods)
        _ (when-not method
            (throw (Error. (format "did not find method %s"
                                   (str method-name)))))
        rettype (.getReturnType method)
        rettype-meta (if (= (str rettype) "void") {} {:tag rettype})
        bodies (map make-a-body methods)
        doc (str/join "\n" (map str methods))]
  (concat ['defn (with-meta method-name (assoc rettype-meta :doc doc))]
          bodies)))

(defmacro defcv-all []
  `(do
     ~@(for [nm (keys zopencv-methods)]
         `(defcv ~(symbol nm)))))

(defcv-all)

(defmacro defcvconst [const-name]
  `(def ~const-name (. com.zenrobotics.zopencv.impl.zopencv ~const-name)))

(defmacro defcvconst-all []
  `(do
     ~@(for [^java.lang.reflect.Field f
             (.getFields com.zenrobotics.zopencv.impl.zopencv)]
         (let [nm (.getName f)]
           `(defcvconst ~(symbol nm))))))

(defcvconst-all)


;;; other stuff

(defn cvZCreateLike
  "Create a new array with the same size as arr, but
  with the given type. No copying."
  [arr t]
  (let [size (cvGetSize arr)]
    (cvCreateMat (.getHeight size)
                 (.getWidth size)
                 t)))