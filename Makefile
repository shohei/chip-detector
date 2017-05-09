PROGRAM=bounding_rect
all:
	g++ -o $(PROGRAM) $(PROGRAM).cpp `pkg-config --libs --cflags opencv`

