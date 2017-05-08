PROGRAM=bounding_rect
all:
	g++ `pkg-config --cflags opencv` ${PROGRAM}.cpp `pkg-config --libs opencv` -o ${PROGRAM} 
