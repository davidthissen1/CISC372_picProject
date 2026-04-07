/*
 * image_pthreads.c
 * Parallelized image convolution using POSIX threads (pthreads).
 *
 * Strategy:
 *   The convolution loop iterates over every (row, pixel, channel) triple
 *   independently — there are NO dependencies between output pixels because
 *   we read from srcImage and write to destImage (separate buffers).
 *   This makes the work embarrassingly parallel.
 *
 *   We divide the image rows evenly among NUM_THREADS threads.  Each thread
 *   owns a contiguous band of rows and computes all pixels/channels in that
 *   band, writing to its own portion of destImage->data.  Because no two
 *   threads ever write to the same memory location there are no race
 *   conditions and no mutex/synchronization is needed during the compute
 *   phase — only a final pthread_join to wait for all threads.
 *
 * Compile:
 *   gcc -O2 -o image_pthreads image_pthreads.c -lpthread
 *
 * Run:
 *   ./image_pthreads <filename> <filter>
 *   where filter is one of: edge sharpen blur gauss emboss identity
 */

#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <pthread.h>
#include "image.h"

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

/* Number of worker threads.  Tune this for the target machine.
   On Darwin's interactive session (4 cores) 4 is a good default. */
#define NUM_THREADS 4

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
/* Per-thread argument bundle                                           */
/* ------------------------------------------------------------------ */
typedef struct {
    Image   *srcImage;   /* read-only source                          */
    Image   *destImage;  /* write destination (non-overlapping bands) */
    Matrix   algorithm;  /* copy of the chosen kernel (read-only)     */
    int      startRow;   /* first row this thread is responsible for  */
    int      endRow;     /* one past the last row  (exclusive)        */
} ThreadArgs;

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
/* Thread worker: process [startRow, endRow) rows                      */
/* ------------------------------------------------------------------ */
void *convolute_rows(void *arg) {
    ThreadArgs *args = (ThreadArgs *)arg;
    Image      *src  = args->srcImage;
    Image      *dst  = args->destImage;
    int row, pix, bit;

    for (row = args->startRow; row < args->endRow; row++) {
        for (pix = 0; pix < src->width; pix++) {
            for (bit = 0; bit < src->bpp; bit++) {
                dst->data[Index(pix, row, src->width, bit, src->bpp)] =
                    getPixelValue(src, pix, row, bit, args->algorithm);
            }
        }
    }
    return NULL;
}

/* ------------------------------------------------------------------ */
/* convolute — launches NUM_THREADS threads, waits for them            */
/* ------------------------------------------------------------------ */
void convolute(Image *srcImage, Image *destImage, Matrix algorithm) {
    pthread_t  threads[NUM_THREADS];
    ThreadArgs args[NUM_THREADS];

    int height       = srcImage->height;
    int rowsPerThread = height / NUM_THREADS;
    int remainder    = height % NUM_THREADS;
    int currentRow   = 0;
    int t;

    for (t = 0; t < NUM_THREADS; t++) {
        args[t].srcImage  = srcImage;
        args[t].destImage = destImage;
        memcpy(args[t].algorithm, algorithm, sizeof(Matrix));
        args[t].startRow  = currentRow;
        /* Distribute the leftover rows one-per-thread among the first threads */
        args[t].endRow    = currentRow + rowsPerThread + (t < remainder ? 1 : 0);
        currentRow        = args[t].endRow;

        pthread_create(&threads[t], NULL, convolute_rows, &args[t]);
    }

    for (t = 0; t < NUM_THREADS; t++) {
        pthread_join(threads[t], NULL);
    }
}

/* ------------------------------------------------------------------ */
/* Helpers (identical to original)                                     */
/* ------------------------------------------------------------------ */
int Usage() {
    printf("Usage: image_pthreads <filename> <type>\n"
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

    printf("Running with %d pthreads on a %dx%d image...\n",
           NUM_THREADS, srcImage.width, srcImage.height);

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
