/*
 * image_openmp.c
 * Parallelized image convolution using OpenMP.
 *
 * Strategy:
 *   The convolution loop over rows is embarrassingly parallel: every output
 *   pixel depends only on a 3×3 neighbourhood in srcImage (read-only during
 *   the whole convolution) and writes to a unique location in destImage.
 *   There are no loop-carried dependencies, so we can safely parallelize the
 *   outer row loop with a single OpenMP pragma.
 *
 *   #pragma omp parallel for schedule(dynamic) instructs OpenMP to distribute
 *   iterations of the row loop across all available threads at runtime.
 *   schedule(dynamic) gives slightly better load balance than the default
 *   static schedule because rows near the image edges do marginally less work
 *   (boundary clamping short-circuits some index calculations).
 *
 *   The number of threads is controlled at runtime via OMP_NUM_THREADS or
 *   the -fopenmp compiler flag's default (usually the number of hardware
 *   cores).  No additional synchronization is needed because each thread
 *   writes to a disjoint set of output indices.
 *
 * Compile:
 *   gcc -O2 -fopenmp -o image_openmp image_openmp.c
 *
 * Run:
 *   OMP_NUM_THREADS=4 ./image_openmp <filename> <filter>
 *   where filter is one of: edge sharpen blur gauss emboss identity
 */

#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <omp.h>
#include "image.h"

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

/* ------------------------------------------------------------------ */
/* Kernel table (same as original)                                      */
/* ------------------------------------------------------------------ */
Matrix algorithms[] = {
    {{0,-1,0},{-1,4,-1},{0,-1,0}},           /* EDGE       */
    {{0,-1,0},{-1,5,-1},{0,-1,0}},           /* SHARPEN    */
    {{1/9.0,1/9.0,1/9.0},{1/9.0,1/9.0,1/9.0},{1/9.0,1/9.0,1/9.0}}, /* BLUR */
    {{1.0/16,1.0/8,1.0/16},{1.0/8,1.0/4,1.0/8},{1.0/16,1.0/8,1.0/16}}, /* GAUSS */
    {{-2,-1,0},{-1,1,1},{0,1,2}},            /* EMBOSS     */
    {{0,0,0},{0,1,0},{0,0,0}}                /* IDENTITY   */
};

/* ------------------------------------------------------------------ */
/* getPixelValue — unchanged from original                             */
/* ------------------------------------------------------------------ */
uint8_t getPixelValue(Image *srcImage, int x, int y, int bit, Matrix algorithm) {
    int px, mx, py, my;
    px = x + 1; py = y + 1; mx = x - 1; my = y - 1;
    if (mx < 0) mx = 0;
    if (my < 0) my = 0;
    if (px >= srcImage->width)  px = srcImage->width  - 1;
    if (py >= srcImage->height) py = srcImage->height - 1;

    uint8_t result =
        algorithm[0][0] * srcImage->data[Index(mx, my, srcImage->width, bit, srcImage->bpp)] +
        algorithm[0][1] * srcImage->data[Index(x,  my, srcImage->width, bit, srcImage->bpp)] +
        algorithm[0][2] * srcImage->data[Index(px, my, srcImage->width, bit, srcImage->bpp)] +
        algorithm[1][0] * srcImage->data[Index(mx, y,  srcImage->width, bit, srcImage->bpp)] +
        algorithm[1][1] * srcImage->data[Index(x,  y,  srcImage->width, bit, srcImage->bpp)] +
        algorithm[1][2] * srcImage->data[Index(px, y,  srcImage->width, bit, srcImage->bpp)] +
        algorithm[2][0] * srcImage->data[Index(mx, py, srcImage->width, bit, srcImage->bpp)] +
        algorithm[2][1] * srcImage->data[Index(x,  py, srcImage->width, bit, srcImage->bpp)] +
        algorithm[2][2] * srcImage->data[Index(px, py, srcImage->width, bit, srcImage->bpp)];
    return result;
}

/* ------------------------------------------------------------------ */
/* convolute — the outer row loop is parallelized with OpenMP          */
/* ------------------------------------------------------------------ */
void convolute(Image *srcImage, Image *destImage, Matrix algorithm) {
    int row, pix, bit;

    /*
     * Parallelize across rows.
     *   - srcImage is read-only  → safe to share among all threads.
     *   - destImage->data: each iteration writes to a unique row-band
     *     → no two threads ever touch the same array element → no race.
     *   - row, pix, bit are declared as private loop variables by default
     *     when inside a parallel for block.
     *   - algorithm is a local copy (passed by value to convolute) → private.
     *   - schedule(dynamic) balances load if row work-units differ slightly.
     */
    #pragma omp parallel for schedule(dynamic) \
        shared(srcImage, destImage, algorithm) \
        private(row, pix, bit) \
        default(none)
    for (row = 0; row < srcImage->height; row++) {
        for (pix = 0; pix < srcImage->width; pix++) {
            for (bit = 0; bit < srcImage->bpp; bit++) {
                destImage->data[Index(pix, row, srcImage->width, bit, srcImage->bpp)] =
                    getPixelValue(srcImage, pix, row, bit, algorithm);
            }
        }
    }
}

/* ------------------------------------------------------------------ */
/* Helpers (identical to original)                                     */
/* ------------------------------------------------------------------ */
int Usage() {
    printf("Usage: image_openmp <filename> <type>\n"
           "\twhere type is one of (edge,sharpen,blur,gauss,emboss,identity)\n");
    return -1;
}

enum KernelTypes GetKernelType(char *type) {
    if      (!strcmp(type, "edge"))    return EDGE;
    else if (!strcmp(type, "sharpen")) return SHARPEN;
    else if (!strcmp(type, "blur"))    return BLUR;
    else if (!strcmp(type, "gauss"))   return GAUSE_BLUR;
    else if (!strcmp(type, "emboss"))  return EMBOSS;
    else                               return IDENTITY;
}

/* ------------------------------------------------------------------ */
/* main                                                                 */
/* ------------------------------------------------------------------ */
int main(int argc, char **argv) {
    long t1, t2;
    t1 = time(NULL);

    stbi_set_flip_vertically_on_load(0);
    if (argc != 3) return Usage();

    char *fileName = argv[1];
    if (!strcmp(argv[1], "pic4.jpg") && !strcmp(argv[2], "gauss")) {
        printf("You have applied a gaussian filter to Gauss which has caused "
               "a tear in the time-space continuum.\n");
    }

    enum KernelTypes type = GetKernelType(argv[2]);

    Image srcImage, destImage;
    srcImage.data = stbi_load(fileName, &srcImage.width, &srcImage.height,
                              &srcImage.bpp, 0);
    if (!srcImage.data) {
        printf("Error loading file %s.\n", fileName);
        return -1;
    }

    destImage.bpp    = srcImage.bpp;
    destImage.height = srcImage.height;
    destImage.width  = srcImage.width;
    destImage.data   = malloc(sizeof(uint8_t) *
                              destImage.width * destImage.bpp * destImage.height);

    printf("Running with %d OpenMP thread(s) on a %dx%d image...\n",
           omp_get_max_threads(), srcImage.width, srcImage.height);

    convolute(&srcImage, &destImage, algorithms[type]);

    stbi_write_png("output.png", destImage.width, destImage.height,
                   destImage.bpp, destImage.data,
                   destImage.bpp * destImage.width);

    stbi_image_free(srcImage.data);
    free(destImage.data);

    t2 = time(NULL);
    printf("Took %ld seconds\n", t2 - t1);
    return 0;
}
