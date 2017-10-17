NUMPY=`python -c 'import numpy; print numpy.get_include()'`
PYROOT=`python -c 'import sys; print sys.prefix'`
VER=`python -c "import sys; print('%s.%s'%(sys.version_info[0],sys.version_info[1]))"`
CC=g++
LIBS= 
#FLAGS= -Wall -DNUMPYCHECK -fPIC
#FLAGS = -Wall -DNDEBUG -O2 -ffast-math -pipe -msse -msse2 -mmmx -mfpmath=sse -fomit-frame-pointer 
#FLAGS = -Wall -DNDEBUG -O2 -ffast-math -fPIC
FLAGS = -DNUMPYCHECK -DNDEBUG -O2 -ffast-math -msse2 -fPIC

.PHONY: all
all: features_pedro_py.so

features_pedro_py.so: features_pedro_py.o
	g++ $^ -shared -o $@ $(LIBS)

features_pedro_py.o: features_pedro_py.cc numpymacros.h
	g++ -c $< $(FLAGS) -I$(NUMPY) -I$(PYROOT)/include/python$(VER) -I../src/ -o $@ 

.PHONY: clean
clean:
	rm -f *.o *.so
