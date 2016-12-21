.PHONY : test
.PHONY : all

objects = bin/main.o bin/simpleImp.o bin/NaiveGPUTranspose.o bin/MatrixTransposeShared.o bin/MatrixTransposeSolveBankConflicts.o bin/MatrixTransposeUnloop.o
test_objects = bin/test.o bin/simpleImp.o bin/NaiveGPUTranspose.o bin/MatrixTransposeShared.o bin/MatrixTransposeSolveBankConflicts.o bin/MatrixTransposeUnloop.o
headers = h/simpleImp.h h/NaiveGPUTranspose.h h/MatrixTransposeShared.h h/MatrixTransposeSolveBankConflicts.h h/MatrixTransposeUnloop.h h/config.h
options = -Wno-deprecated-gpu-targets -I h/#

bin/main : $(objects)
	nvcc $(options) $(objects) -o bin/main

bin/main.o : src/main.cu $(headers)
	nvcc -c $(options) src/main.cu -o bin/main.o

bin/simpleImp.o : src/simpleImp.cpp h/simpleImp.h
	nvcc -c $(options) src/simpleImp.cpp -o bin/simpleImp.o

bin/NaiveGPUTranspose.o : src/NaiveGPUTranspose.cu h/NaiveGPUTranspose.h
	nvcc -c $(options) src/NaiveGPUTranspose.cu -o bin/NaiveGPUTranspose.o

bin/MatrixTransposeShared.o : src/MatrixTransposeShared.cu h/MatrixTransposeShared.h h/config.h
	nvcc -c $(options) src/MatrixTransposeShared.cu -o bin/MatrixTransposeShared.o

bin/MatrixTransposeSolveBankConflicts.o : src/MatrixTransposeSolveBankConflicts.cu h/MatrixTransposeSolveBankConflicts.h h/config.h
	nvcc -c $(options) src/MatrixTransposeSolveBankConflicts.cu -o bin/MatrixTransposeSolveBankConflicts.o

bin/MatrixTransposeUnloop.o : src/MatrixTransposeUnloop.cu h/MatrixTransposeUnloop.h h/config.h
	nvcc -c $(options) src/MatrixTransposeUnloop.cu -o bin/MatrixTransposeUnloop.o

clear :
	@rm -f bin/*

test :
	@nvcc -c $(options) src/test.cu -o bin/test.o
	@nvcc $(options) $(test_objects) -o bin/test
	@echo "Start test..."
	@echo "*********************************************"
	@bin/test
	@echo "*********************************************"
	@echo "Finish."
