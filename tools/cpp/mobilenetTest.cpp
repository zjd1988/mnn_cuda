//
//  mobilenetV1Test.cpp
//  MNN
//
//  Created by MNN on 2018/05/14.
//  Copyright © 2018, Alibaba Group Holding Limited
//
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#include "stb_image_write.h"

#include "mtcnn_net.hpp"

int main(int argc, const char* argv[]) {

    if (argc < 2) {
        MNN_PRINT("Usage: ./mobilenetTest.out input.jpg [word.txt]\n");
        return 0;
    }
    MTCNN *mtcnn = MTCNN::get_instance();
    //get test img data
    int imageChannel, imageWidth, imageHeight;
    unsigned char* inputImage = stbi_load(argv[1], &imageWidth,
                                            &imageHeight, &imageChannel, 4);
    ImageData img;
    // img.channels = imageChannel;
    img.channels = 4;
    img.data     = inputImage;
    img.height   = imageHeight;
    img.width    = imageWidth;
    img.channel_type = Channel_BGRA;//0 for BGRA
    std::vector<FaceInfo> face = mtcnn->runFaceDect(img);
    std::vector<FacePoint> points = mtcnn->runDetecPoints(face[0]);
    ////for test
    // points[0].x = 249.64622390270233f;
    // points[0].y = 158.15639609098434f;

    // points[1].x = 301.22084903717041f;
    // points[1].y = 157.0186402797699f;

    // points[2].x = 281.62730252742767f;
    // points[2].y = 182.11544513702393f;

    // points[3].x = 257.83306503295898f;
    // points[3].y = 213.7911319732666f;

    // points[4].x = 294.91109192371368f;
    // points[4].y = 212.42057943344116f;
    std::vector<float>    feature = mtcnn->getFaceFeature(points);
    stbi_image_free(inputImage);
    printf("done!\n");

    return 0;
}