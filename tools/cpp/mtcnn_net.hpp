#ifndef MTCNN_NET_HPP
#define MTCNN_NET_HPP
#include <MNN/ImageProcess.hpp>
#include <MNN/Interpreter.hpp>

#include <stdio.h>
#include <algorithm>
#include <fstream>
#include <functional>
#include <memory>
#include <sstream>
#include <vector>
using namespace MNN;
using namespace MNN::CV;

typedef struct RegRect
{
    int32_t x;
    int32_t y;
    int32_t width;
    int32_t height;
    int32_t scale;
    double conf;
    RegRect( int32_t x_ = 0, int32_t y_ = 0, int32_t w_ = 0, int32_t h_ = 0, int32_t s_ = 0, double c_ = 0 )
        : x( x_ ), y( y_ ), width( w_ ), height( h_ ), scale( s_ ), conf( c_ ) {
    }
}RegRect;

typedef struct FaceInfo
{
    RegRect pos;
    float score;
}FaceInfo;


typedef struct FacePoint
{
    float x;
    float y;
}FacePoint;


typedef enum ChannelType{
    Channel_BGRA,
    Channel_BGR,
    Channel_RGBA,
    Channel_RGB,
    Channel_Max
}ChannelType;

typedef struct ImageData
{
    int width;
    int height;
    int channels;
    ChannelType channel_type;  //0 for bgra; 1 for bgr; 2 for rgba; 3 for rgb
    unsigned char *data;
}ImageData;


class MTCNN {
public:
    std::vector<FaceInfo> runFaceDect(ImageData imageInfo);
    std::vector<FacePoint> runDetecPoints(FaceInfo faceInfo);
    std::vector<float>     getFaceFeature(std::vector<FacePoint> &facePoint);
    float                 computeSimilarity(std::vector<float> feature1, std::vector<float> feature2);
    MTCNN();
    ~MTCNN();
private:
    //1---------------------face detect
    std::vector<RegRect> runPnet();
    std::vector<RegRect> runRnet(std::vector<RegRect> &winList);
    std::vector<RegRect> runOnet(std::vector<RegRect> &winList);
    void transWindowToFace(std::vector<RegRect> &winList);
    static bool CompareWin( const RegRect &w1, const RegRect &w2 );
    float IoU( const RegRect &w1, const RegRect &w2 );
    bool Legal( int x, int y, const int img_width, const int img_height );
    std::vector<RegRect> NMS( std::vector<RegRect> &winList, bool local, float threshold );
    void imgPreprocess(const unsigned char* inputImage, int imageWidth, int imageHeight, 
                int netInputChannel, int netInputWidth, int netInputHeight, Tensor &inputTensorUser);
    void netInputPreprocess(std::vector<RegRect> &winList, Tensor &inputTensorUser, int netInputChannel, int netInputWidth, int netInputHeight);
    void netInputPreprocess(RegRect &winList, Tensor &inputTensorUser, int netInputChannel, int netInputWidth, int netInputHeight);
    void cropImage(RegRect *cropRect, unsigned char* cropData);
    //face detect net config
    const char* pnetName = "/home/xj-zjd/work_space/self_work/seetaface2_code/MNN/build/pnet.mnn";
    const char* rnetName = "/home/xj-zjd/work_space/self_work/seetaface2_code/MNN/build/rnet.mnn";
    const char* onetName = "/home/xj-zjd/work_space/self_work/seetaface2_code/MNN/build/onet.mnn";
    float thresh[3] = {0.7, 0.7, 0.9};
    std::shared_ptr<Interpreter> pnet;
    std::shared_ptr<Interpreter> rnet;
    std::shared_ptr<Interpreter> onet;
    Session* pnetSession;
    Session* rnetSession;
    Session* onetSession;
    std::vector<Tensor*> pnetInputTensors;
    std::vector<Tensor*> pnetOutputTensors;
    std::vector<Tensor*> rnetInputTensors;
    std::vector<Tensor*> rnetOutputTensors;
    std::vector<Tensor*> onetInputTensors;
    std::vector<Tensor*> onetOutputTensors;
    unsigned char*inputImageData = NULL;
    int inputImageWidth;
    int inputImageHeight;
    int inputImageChannel;
    ChannelType inputImageChannelType;
    
    int pnetInputBatch;
    int pnetInputWidth;
    int pnetInputHeight;
    int pnetInputChannel;

    int rnetInputBatch = 10;
    int rnetInputChannel = 3;
    int rnetInputWidth = 24;
    int rnetInputHeight = 24;
    

    int onetInputBatch = 5;
    int onetInputChannel = 3;
    int onetInputWidth = 48;
    int onetInputHeight = 48;
    
    //pnet config
    int PNET_SIZE = 12;
    float PNET_THRESH = 0.7f;
    int   PNET_STRIDE = 4;
    float PNET_SCALE  = 1.414f;
    float PNET_NMS_THRESH = 0.8f;
    //rnet config
    int RNET_MAX_INPUT_RECT_NUM = 10;
    float RNET_THRESH = 0.7f;
    float RNET_NMS_THRESH = 0.8f;
    //onet config
    int ONET_MAX_INPUT_RECT_NUM = 5;
    float ONET_THRESH = 0.85f;
    float ONET_NMS_THRESH = 0.3f;

    //detect face result
    std::vector<FaceInfo> detectFaceInfo;


    //face landmark 5 points
    void shapeIndexPatch(float*input1, std::vector<int> dim1, float*input2, std::vector<int> dim2, float* output);
    const char* landmarkNetName1 = "/home/xj-zjd/work_space/self_work/seetaface2_code/MNN/build/Points5_Net1.mnn";
    const char* landmarkNetName2 = "/home/xj-zjd/work_space/self_work/seetaface2_code/MNN/build/Points5_Net2.mnn";
    std::shared_ptr<Interpreter> landmarkNet1;
    std::shared_ptr<Interpreter> landmarkNet2;
    Session* landmarkNetSession1;
    Session* landmarkNetSession2;
    std::vector<Tensor*> landmarkInputTensors1;
    std::vector<Tensor*> landmarkOutputTensors1;
    std::vector<Tensor*> landmarkInputTensors2;
    std::vector<Tensor*> landmarkOutputTensors2;
    std::vector<FacePoint> detectFacePoints;
    int pointNum = 5;
    int originPatch = 15;
    int landmarkNet1InputBatch = 1;
    int landmarkNet1InputChannel = 1;
    int landmarkNet1InputWidth = 112;
    int landmarkNet1InputHeight = 112;
    int landmarkNet2InputBatch = 1;
    int landmarkNet2InputChannel = 8;
    int landmarkNet2InputWidth = 4;
    int landmarkNet2InputHeight = 20;
    //face recognization
    int CROP_FACE_WIDTH = 256;
    int CROP_FACE_HEIGHT = 256;
    void cropImage(unsigned char* srcData, int srcWidth, int srcHeight, int srcChannel,
    RegRect *cropRect, unsigned char* cropData);
    void faceImgPreprocess(const unsigned char* inputImage, int imageWidth, int imageHeight, 
                int netInputChannel, int netInputWidth, int netInputHeight, Tensor &inputTensorUser);
    void cropFace(std::vector<FacePoint> &facePoint, int height, int width, int channels, uint8_t* faceData);
    void ExtractFeature(uint8_t *faceData, int width, int height);
    const char* faceRecogNetName = "/home/xj-zjd/work_space/self_work/seetaface2_code/MNN/build/FaceRecognize_Net.mnn";
    std::shared_ptr<Interpreter> faceRecogNet;
    Session* faceRecogNetSession;
    std::vector<Tensor*> faceRecogInputTensors;
    std::vector<Tensor*> faceRecogOutputTensors;
    std::vector<float>   faceFeature;
    int faceRecogNetInputBatch = 1;
    int faceRecogNetInputChannel = 3;
    int faceRecogNetInputWidth = 248;
    int faceRecogNetInputHeight = 248;

    static MTCNN* mtcnn;    
public:
    static MTCNN* get_instance();

};
#endif