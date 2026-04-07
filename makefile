# Makefile for CISC372 image convolution project

# ── Serial (original) ────────────────────────────────────────────────
image: image.c image.h
	gcc -O2 -Wall -g -o image image.c -lm

# ── pthreads ─────────────────────────────────────────────────────────
image_pthreads: image_pthreads.c image.h
	gcc -O2 -Wall -g -o image_pthreads image_pthreads.c -lpthread -lm

# ── OpenMP ───────────────────────────────────────────────────────────
image_openmp: image_openmp.c image.h
	gcc -O2 -Wall -g -fopenmp -o image_openmp image_openmp.c -lm

all: image image_pthreads image_openmp

clean:
	rm -f image image_pthreads image_openmp output.png

