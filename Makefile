###########################################
### Fill in locations of dependencies here

# you can leave ccache blank if you don't want/have it
CCACHE:=
SWIG:=$(CCACHE) swig2.0
CXX:=g++

JAVAC:=javac
JAVA:=java

# the directory where you have your jni.h
JAVA_INCLUDE=/usr/lib/jvm/java-6-openjdk/include

# the path to a clojure jar, used for starting the repl
CLOJURE:=path/to/clojure-1.5.1.jar

# path to opencv headers
# should contain files like opencv/cv.h and opencv2/core/core.hpp
OPENCV_INCLUDE:=path/to/2.3.1/include

# if your opencv .so's are in a nonstandard location, input that here
LIBDIR:=

### You can stop filling in
############################################

# configuration
OUT:=build
IMPL_PACKAGE:=com.zenrobotics.zopencv.impl

# don't touch
IMPL_DIR:=$(OUT)/$(subst .,/,$(IMPL_PACKAGE))
JAVAFILES:=$(shell find src -name *java)
LDFLAGS:=$(LIBDIR:%=-L%) -lopencv_core -lopencv_calib3d -lopencv_imgproc -lopencv_contrib -lopencv_features2d -lopencv_highgui -lopencv_ml -lopencv_legacy -lopencv_video -lopencv_objdetect
CFLAGS:=-I$(OPENCV_INCLUDE) -I$(OPENCV_INCLUDE)/opencv -I$(OPENCV_INCLUDE)/opencv2 -Isrc -I$(JAVA_INCLUDE)
SWIGFLAGS:=$(CFLAGS)

$(info LDFLAGS $(LDFLAGS))

# rules
all: repl

clean:
	rm -fr $(OUT)

$(OUT)/zopencv_wrap.cxx: src/zopencv.i
	mkdir -p $(OUT)
	mkdir -p $(IMPL_DIR)
	$(SWIG) $(SWIGFLAGS) -c++ -java -package $(IMPL_PACKAGE) -outdir $(IMPL_DIR) -o $@ $<

$(OUT)/libzopencv.so: $(OUT)/zopencv_wrap.cxx
	mkdir -p $(OUT)
	$(CXX) -shared -fPIC -o $@ $< $(CFLAGS) $(LDFLAGS)

$(OUT)/compile_java: $(OUT)/libzopencv.so # also implicitly depend on swig producing the java files
	mkdir -p $(OUT)
	$(JAVAC) -d $(OUT) $(IMPL_DIR)/*java $(JAVAFILES)
	touch $@

.PHONY: repl
repl: $(OUT)/compile_java
	$(LIBDIR:%=LD_LIBRARY_PATH=%) $(JAVA) -cp $(OUT):src:$(CLOJURE) -Djava.library.path=$(OUT):$(LIBDIR) clojure.main