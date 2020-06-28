#include "mtcnn_net.hpp"
#include "common_alignment.hpp"

class HypeShape {
public:
    using self = HypeShape;
    using T = int32_t;

    explicit HypeShape( const std::vector<int> &shape )
        : m_shape( shape ) {
        // update weights
        if( m_shape.empty() ) return;
        m_weights.resize( m_shape.size() );
        auto size = m_shape.size();
        auto weight_it = m_weights.rbegin();
        auto shape_it = m_shape.rbegin();
        *weight_it++ = *shape_it++;
        for( size_t times = size - 1; times; --times ) {
            *weight_it = *( weight_it - 1 ) * *shape_it;
            ++weight_it;
            ++shape_it;
        }
    }

    T to_index( const std::initializer_list<T> &coordinate ) const {
        if( coordinate.size() == 0 ) return 0;
        auto size = coordinate.size();
        auto weight_it = m_weights.end() - size + 1;
        auto coordinate_it = coordinate.begin();
        T index = 0;
        for( size_t times = size - 1; times; --times ) {
            index += *weight_it * *coordinate_it;
            ++weight_it;
            ++coordinate_it;
        }
        index += *coordinate_it;
        return index;
    }

    T to_index( const std::vector<T> &coordinate ) const {
        if( coordinate.empty() ) return 0;
        auto size = coordinate.size();
        auto weight_it = m_weights.end() - size + 1;
        auto coordinate_it = coordinate.begin();
        T index = 0;
        for( size_t times = size - 1; times; --times ) {
            index += *weight_it * *coordinate_it;
            ++weight_it;
            ++coordinate_it;
        }
        index += *coordinate_it;
        return index;
    }

    T to_index( int arg0 ) {
        return arg0;
    }


#define LOOP_HEAD(n) const size_t size = (n); auto weight_it = m_weights.end() - size + 1; T index = 0;
#define LOOP_ON(i) index += *weight_it * arg##i; ++weight_it;
#define LOOP_END(i) index += arg##i; return index;

    T to_index( int arg0, int arg1 ) {
        LOOP_HEAD( 2 )
        LOOP_ON( 0 )
        LOOP_END( 1 )
    }

    T to_index( int arg0, int arg1, int arg2 ) {
        LOOP_HEAD( 3 )
        LOOP_ON( 0 ) LOOP_ON( 1 )
        LOOP_END( 2 )
    }

    T to_index( int arg0, int arg1, int arg2, int arg3 ) {
        LOOP_HEAD( 4 )
        LOOP_ON( 0 ) LOOP_ON( 1 ) LOOP_ON( 2 )
        LOOP_END( 3 )
    }

    T to_index( int arg0, int arg1, int arg2, int arg3, int arg4 ) {
        LOOP_HEAD( 5 )
        LOOP_ON( 0 ) LOOP_ON( 1 ) LOOP_ON( 2 ) LOOP_ON( 3 )
        LOOP_END( 4 )
    }

    T to_index( int arg0, int arg1, int arg2, int arg3, int arg4,
                int arg5 ) {
        LOOP_HEAD( 6 )
        LOOP_ON( 0 ) LOOP_ON( 1 ) LOOP_ON( 2 ) LOOP_ON( 3 ) LOOP_ON( 4 )
        LOOP_END( 5 )
    }

    ///
    T to_index( int arg0, int arg1, int arg2, int arg3, int arg4,
                int arg5, int arg6 ) {
        LOOP_HEAD( 7 )
        LOOP_ON( 0 ) LOOP_ON( 1 ) LOOP_ON( 2 ) LOOP_ON( 3 ) LOOP_ON( 4 )
        LOOP_ON( 5 )
        LOOP_END( 6 )
    }

    T to_index( int arg0, int arg1, int arg2, int arg3, int arg4,
                int arg5, int arg6, int arg7 ) {
        LOOP_HEAD( 8 )
        LOOP_ON( 0 ) LOOP_ON( 1 ) LOOP_ON( 2 ) LOOP_ON( 3 ) LOOP_ON( 4 )
        LOOP_ON( 5 ) LOOP_ON( 6 )
        LOOP_END( 7 )
    }

    T to_index( int arg0, int arg1, int arg2, int arg3, int arg4,
                int arg5, int arg6, int arg7, int arg8 ) {
        LOOP_HEAD( 9 )
        LOOP_ON( 0 ) LOOP_ON( 1 ) LOOP_ON( 2 ) LOOP_ON( 3 ) LOOP_ON( 4 )
        LOOP_ON( 5 ) LOOP_ON( 6 ) LOOP_ON( 7 )
        LOOP_END( 8 )
    }

    T to_index( int arg0, int arg1, int arg2, int arg3, int arg4,
                int arg5, int arg6, int arg7, int arg8, int arg9 ) {
        LOOP_HEAD( 10 )
        LOOP_ON( 0 ) LOOP_ON( 1 ) LOOP_ON( 2 ) LOOP_ON( 3 ) LOOP_ON( 4 )
        LOOP_ON( 5 ) LOOP_ON( 6 ) LOOP_ON( 7 ) LOOP_ON( 8 )
        LOOP_END( 9 )
    }

#undef LOOP_HEAD
#undef LOOP_ON
#undef LOOP_END

    std::vector<T> to_coordinate( T index ) const {
        if( m_shape.empty() )
            return std::vector<T>();
        std::vector<T> coordinate( m_shape.size() );
        to_coordinate( index, coordinate );
        return std::move( coordinate );
    }

    void to_coordinate( T index, std::vector<T> &coordinate ) const {
        if( m_shape.empty() ) {
            coordinate.clear();
            return;
        }
        coordinate.resize( m_shape.size() );
        auto size = m_shape.size();
        auto weight_it = m_weights.begin() + 1;
        auto coordinate_it = coordinate.begin();
        for( size_t times = size - 1; times; --times ) {
            *coordinate_it = index / *weight_it;
            index %= *weight_it;
            ++weight_it;
            ++coordinate_it;
        }
        *coordinate_it = index;
    }

    T count() const {
        return m_weights.empty() ? 1 : m_weights[0];
    }

    T weight( size_t i ) const {
        return m_weights[i];
    };

    const std::vector<T> &weight() const {
        return m_weights;
    };

    T shape( size_t i ) const {
        return m_shape[i];
    };

    const std::vector<T> &shape() const {
        return m_shape;
    };

    explicit operator std::vector<int>() const {
        return m_shape;
    }

private:
    std::vector<int32_t> m_shape;
    std::vector<T> m_weights;

public:
    HypeShape( const self &other ) = default;
    HypeShape &operator=( const self &other ) = default;

    HypeShape( self &&other ) {
        *this = std::move( other );
    }
    HypeShape &operator=( self &&other ) {
#define MOVE_MEMBER(member) this->member = std::move(other.member)
        MOVE_MEMBER( m_shape );
        MOVE_MEMBER( m_weights );
#undef MOVE_MEMBER
        return *this;
    }
};

bool MTCNN::CompareWin( const RegRect &w1, const RegRect &w2 )
{
    return w1.conf > w2.conf;
}

bool MTCNN::Legal( int x, int y, const int img_width, const int img_height ) {
    if( x >= 0 && x < img_width && y >= 0 && y < img_height )
        return true;
    else
        return false;
}

float MTCNN::IoU( const RegRect &w1, const RegRect &w2 ) {
    int xOverlap = std::max( 0, std::min( w1.x + w1.width - 1, w2.x + w2.width - 1 ) - std::max( w1.x, w2.x ) + 1 );
    int yOverlap = std::max( 0, std::min( w1.y + w1.height - 1, w2.y + w2.height - 1 ) - std::max( w1.y, w2.y ) + 1 );
    int intersection = xOverlap * yOverlap;
    int unio = w1.width * w1.height + w2.width * w2.height - intersection;
    return float( intersection ) / unio;
}

std::vector<RegRect> MTCNN::NMS( std::vector<RegRect> &winList, bool local, float threshold )
{
    if( winList.size() == 0 )
        return winList;
    std::sort( winList.begin(), winList.end(), CompareWin );
    std::vector<bool> flag( winList.size(), false );
    for( size_t i = 0; i < winList.size(); i++ )
    {
        if( flag[i] )
            continue;
        for( size_t j = i + 1; j < winList.size(); j++ )
        {
            if( local && winList[i].scale != winList[j].scale )
                continue;
            if( IoU( winList[i], winList[j] ) > threshold )
                flag[j] = true;
        }
    }
    std::vector<RegRect> ret;
    for( size_t i = 0; i < winList.size(); i++ )
    {
        if( !flag[i] ) ret.push_back( winList[i] );
    }
    return ret;
}


void MTCNN::imgPreprocess(const unsigned char* inputImage, int imageWidth, int imageHeight, 
                int netInputChannel, int netInputWidth, int netInputHeight, Tensor &inputTensorUser)
{
    Matrix trans;
    trans.setScale(1.0 / imageWidth, 1.0 / imageHeight);
    trans.postRotate(0, 0.5f, 0.5f);
    trans.postScale(netInputWidth, netInputHeight);
    trans.invert(&trans);

    ImageProcess::Config config;
    config.filterType = BILINEAR;

    if(inputImageChannelType == Channel_BGRA)
        config.sourceFormat = BGRA;
    else if(inputImageChannelType == Channel_RGBA)
        config.sourceFormat = RGBA;
    else if(inputImageChannelType == Channel_BGR)
        config.sourceFormat = BGR;
    else
        config.sourceFormat = RGB;

    float mean[3]     = {104.f, 117.f, 123.f};
    float normals[3] = {1.0f, 1.0f, 1.0f};
    if(netInputChannel == 3)
        config.destFormat = RGB;
    else
    {
        config.destFormat = GRAY;
        mean[0] = 0.0f;
        mean[1] = 0.0f;
        mean[2] = 0.0f;
    }
    ::memcpy(config.mean, mean, sizeof(mean));
    ::memcpy(config.normal, normals, sizeof(normals));

    std::shared_ptr<ImageProcess> pretreat(ImageProcess::create(config));
    pretreat->setMatrix(trans);
    pretreat->convert(inputImage, imageWidth, imageHeight, 0, &inputTensorUser);
    return;
}



void MTCNN::faceImgPreprocess(const unsigned char* inputImage, int imageWidth, int imageHeight, 
                int netInputChannel, int netInputWidth, int netInputHeight, Tensor &inputTensorUser)
{
    Matrix trans;
    trans.setScale(1.0 / imageWidth, 1.0 / imageHeight);
    trans.postRotate(0, 0.5f, 0.5f);
    trans.postScale(netInputWidth, netInputHeight);
    trans.invert(&trans);

    ImageProcess::Config config;
    config.filterType = BILINEAR;

    if(inputImageChannelType == Channel_BGRA)
        config.sourceFormat = BGRA;
    else if(inputImageChannelType == Channel_RGBA)
        config.sourceFormat = RGBA;
    else if(inputImageChannelType == Channel_BGR)
        config.sourceFormat = BGR;
    else
        config.sourceFormat = RGB;

    float mean[3]     = {0.0f, 0.0f, 0.0f};
    float normals[3] = {0.00392156886f, 0.00392156886f, 0.00392156886f};
    if(netInputChannel == 3)
        config.destFormat = BGR;
    else
    {
        config.destFormat = GRAY;
        mean[0] = 0.0f;
        mean[1] = 0.0f;
        mean[2] = 0.0f;
    }
    ::memcpy(config.mean, mean, sizeof(mean));
    ::memcpy(config.normal, normals, sizeof(normals));

    std::shared_ptr<ImageProcess> pretreat(ImageProcess::create(config));
    pretreat->setMatrix(trans);
    pretreat->convert(inputImage, imageWidth, imageHeight, 0, &inputTensorUser);
    return;
}


std::vector<RegRect> MTCNN::runPnet()
{
    //run
    pnetInputHeight = pnetInputTensors[0]->height();
    pnetInputWidth  = pnetInputTensors[0]->width();
    pnetInputChannel = pnetInputTensors[0]->channel();
    int resizeHeight = pnetInputHeight;
    int resizeWidth  = pnetInputWidth;
    int count_scale = 0;
    float cur_scale = 0.0;
    std::vector<RegRect> winList;
    while(std::min( resizeWidth, resizeHeight ) >= PNET_SIZE)
    {
        std::vector<int> inputDims = {1, 3, resizeHeight, resizeWidth};
        pnet->resizeTensor(pnetInputTensors[0], inputDims);
        pnet->resizeSession(pnetSession);
        //image preproccess
        imgPreprocess(inputImageData, inputImageWidth, inputImageHeight, pnetInputChannel, resizeWidth, resizeHeight, *pnetInputTensors[0]);
        float cur_scale = float( inputImageHeight ) / resizeHeight;
        
        // inputTensor->copyFromHostTensor(&inputTensorUser);
        pnet->runSession(pnetSession);
        Tensor outputTensorUser1(pnetOutputTensors[0], pnetOutputTensors[0]->getDimensionType());
        Tensor outputTensorUser2(pnetOutputTensors[1], pnetOutputTensors[1]->getDimensionType());
        pnetOutputTensors[0]->copyToHostTensor(&outputTensorUser1);
        pnetOutputTensors[1]->copyToHostTensor(&outputTensorUser2);

        // std::vector<int> shape = outputTensorUser1.shape();
        // float *data = outputTensorUser1.host<float>();
        // for(int i = 0; i < shape[1]; i++)
        // {
        //     for(int j = 0; j < shape[2]; j++)
        //     {
        //         for(int k = 0; k < shape[3]; k++)
        //         {
        //             printf("%f ", data[i*shape[2]*shape[3] + j*shape[3] +k]);
        //         }
        //         printf("\n");
        //     }
        //     printf("\n");
        // }

        auto cls_shape = outputTensorUser1.shape();
        auto reg_shape = outputTensorUser2.shape();
        int n_reg = reg_shape[0];
        int c_reg = reg_shape[1];
        int h_reg = reg_shape[2];
        int w_reg = reg_shape[3];
        int n_cls = cls_shape[0];
        int c_cls = cls_shape[1];
        int h_cls = cls_shape[2];
        int w_cls = cls_shape[3];
        float *cls_data = outputTensorUser1.host<float>();
        float *reg_data = outputTensorUser2.host<float>();

        float w = PNET_SIZE * cur_scale;
        for( int i = 0; i < h_reg; i++ )
        {
            for( int j = 0; j < w_reg; j++ )
            {
                if( cls_data[( 1 * h_reg + i ) * w_reg + j] > PNET_THRESH )
                {
                    float sn = reg_data[( 0 * h_reg + i ) * w_reg + j];
                    float xn = reg_data[( 1 * h_reg + i ) * w_reg + j];
                    float yn = reg_data[( 2 * h_reg + i ) * w_reg + j];

                    int rx, ry, rw;


                    int crop_x = int( j * cur_scale * PNET_STRIDE );
                    int crop_y = int( i * cur_scale * PNET_STRIDE );
                    int crop_w = int( w );
                    rx = int( crop_x - 0.5 * sn * crop_w + crop_w * sn * xn + 0.5 * crop_w );
                    ry = int( crop_y - 0.5 * sn * crop_w + crop_w * sn * yn + 0.5 * crop_w );
                    rw = int( sn * crop_w );


                    if( Legal( rx, ry, inputImageWidth, inputImageHeight ) && Legal( rx + rw - 1, ry + rw - 1, inputImageWidth, inputImageHeight ) )
                    {
                        winList.push_back( RegRect( rx, ry, rw, rw, count_scale, cls_data[( 1 * h_reg + i ) * w_reg + j] ) );
                    }
                }
            }
        }

        resizeHeight = int( resizeHeight / PNET_SCALE );
        resizeWidth = int( resizeWidth / PNET_SCALE );
        count_scale++;
    }
    winList = NMS(winList, true, PNET_NMS_THRESH);
    return winList;
}

void MTCNN::cropImage(RegRect *cropRect, unsigned char* cropData)
{

    int count = 0;
    int channelNum = 3;
    if (inputImageChannelType == Channel_BGRA || inputImageChannelType == Channel_RGBA)
        channelNum = 4;
    int heightStride = inputImageWidth * channelNum;
    int copyLen = cropRect->width * channelNum;        
    int rowOffset = cropRect->x * channelNum;
    for(int i = cropRect->y; i < cropRect->y + cropRect->height; i++)
    {
        int offset = count * copyLen;
        memcpy(cropData + offset, inputImageData + i * heightStride + rowOffset, copyLen);
        count++;
    }
}

void MTCNN::cropImage(unsigned char* srcData, int srcWidth, int srcHeight, int srcChannel,
    RegRect *cropRect, unsigned char* cropData)
{
    int count = 0;
    int channelNum = srcChannel;
    int heightStride = srcWidth * channelNum;
    int copyLen = cropRect->width * channelNum;
    int rowOffset = cropRect->x * channelNum;
    for(int i = cropRect->y; i < cropRect->y + cropRect->height; i++)
    {
        int offset = count * copyLen;
        memcpy(cropData + offset, srcData + i * heightStride + rowOffset, copyLen);
        count++;
    }
}

void MTCNN::netInputPreprocess(std::vector<RegRect> &winList, Tensor &inputTensorUser, int netInputChannel, 
            int netInputWidth, int netInputHeight)
{
    int channel_num = 0;
    auto shape = inputTensorUser.shape();
    shape[0] = 1;
    if(inputImageChannelType == Channel_BGRA || inputImageChannelType == Channel_RGBA)
        channel_num = 4;
    else
        channel_num = 3;
    
    for(int i = 0; i < winList.size(); i++)
    {
        float* startAddr = inputTensorUser.host<float>() + i * inputTensorUser.stride(0);
        std::shared_ptr<Tensor> tempTensor;
        tempTensor.reset(Tensor::create(shape, inputTensorUser.getType(), startAddr, inputTensorUser.getDimensionType()));
        uint8_t *cropData = new uint8_t[1 * channel_num * winList[i].width * winList[i].height];
        cropImage(&winList[i], cropData);
        imgPreprocess(cropData, winList[i].width, winList[i].height, netInputChannel, netInputWidth, netInputHeight, *(tempTensor.get()));
        delete cropData;
    }
    return;
}


void MTCNN::netInputPreprocess(RegRect &winList, Tensor &inputTensorUser, int netInputChannel, 
            int netInputWidth, int netInputHeight)
{
    int channel_num = 0;
    auto shape = inputTensorUser.shape();
    shape[0] = 1;
    if(inputImageChannelType == Channel_BGRA || inputImageChannelType == Channel_RGBA)
        channel_num = 4;
    else
        channel_num = 3;
    
    float* startAddr = inputTensorUser.host<float>();
    std::shared_ptr<Tensor> tempTensor;
    tempTensor.reset(Tensor::create(shape, inputTensorUser.getType(), startAddr, inputTensorUser.getDimensionType()));
    uint8_t *cropData = new uint8_t[1 * channel_num * winList.width * winList.height];
    cropImage(&winList, cropData);
    imgPreprocess(cropData, winList.width, winList.height, netInputChannel, netInputWidth, netInputHeight, *(tempTensor.get()));
    delete cropData;
    return;
}

std::vector<RegRect> MTCNN::runRnet(std::vector<RegRect> &winList)
{
    if( winList.size() == 0 )
        return winList;
    std::vector<RegRect> ret;
    while( winList.size() )
    {
        std::vector<RegRect> tmp_winList;
        int count = winList.size() < rnetInputBatch ? winList.size():rnetInputBatch;
        while( winList.size() && count )
        {
            tmp_winList.push_back( winList.back() );
            winList.pop_back();
            count--;
        }
        //for test begin
        tmp_winList[0].x = 218;
        tmp_winList[0].y = 64;
        tmp_winList[1].x = 214;
        tmp_winList[1].y = 82;
        tmp_winList[2].x = 111;
        tmp_winList[2].y = 192;
        tmp_winList[3].x = 162;
        tmp_winList[3].y = 5;
        tmp_winList[4].x = 168;
        tmp_winList[4].y = 79;
        tmp_winList[5].x = 107;
        tmp_winList[5].y = 223;
        tmp_winList[6].x = 192;
        tmp_winList[6].y = 141;
        tmp_winList[7].x = 197;
        tmp_winList[7].y = 111;     
        tmp_winList[8].x = 186;
        tmp_winList[8].y = 103;
        tmp_winList[9].x = 216;
        tmp_winList[9].y = 123;                                                                 
        //
        
        Tensor inputTensorUser1(rnetInputTensors[0], rnetInputTensors[0]->getDimensionType());
        netInputPreprocess(tmp_winList, inputTensorUser1, rnetInputChannel, rnetInputWidth, rnetInputHeight);
        rnetInputTensors[0]->copyFromHostTensor(&inputTensorUser1);
        rnet->runSession(rnetSession);
        Tensor outputTensorUser1(rnetOutputTensors[0], rnetOutputTensors[0]->getDimensionType());
        Tensor outputTensorUser2(rnetOutputTensors[1], rnetOutputTensors[1]->getDimensionType());
        rnetOutputTensors[0]->copyToHostTensor(&outputTensorUser1);
        rnetOutputTensors[1]->copyToHostTensor(&outputTensorUser2);


        Tensor* temp = rnet->getSessionOutput(rnetSession, "output1");
        Tensor tempUser(temp, temp->getDimensionType());
        temp->copyToHostTensor(&tempUser);
        std::vector<int> shape = tempUser.shape();
        float *data = tempUser.host<float>();
        // for(int batch = 0; batch < shape[0]; batch++)
        // {
        //     int batchStride = batch * shape[1] * shape[2] * shape[3];
        //     for(int i = 0; i < shape[1]; i++)
        //     {
        //         for(int j = 0; j < shape[2]; j++)
        //         {
        //             for(int k = 0; k < shape[3]; k++)
        //             {
        //                 printf("%f ", data[batchStride + i*shape[2]*shape[3] + j*shape[3] +k]);
        //             }
        //             printf("\n");
        //         }
        //         printf("\n");
        //     }
        //     printf("\n");
        // }
        // for(int batch = 0; batch < shape[0]; batch++)
        // {
        //     int batchStride = batch * shape[1];
        //     for(int i = 0; i < shape[1]; i++)
        //     {
        //         printf("%f ", data[batchStride + i]);
        //     }
        //     printf("\n");
        // }        


        auto cls_shape = outputTensorUser1.shape();
        auto reg_shape = outputTensorUser2.shape();
        int n_reg = reg_shape[0];
        int c_reg = reg_shape[1];
        int h_reg = 1;
        int w_reg = 1;
        int n_cls = cls_shape[0];
        int c_cls = cls_shape[1];
        int h_cls = 1;
        int w_cls = 1;

        float *cls_data = outputTensorUser1.host<float>();
        float *reg_data = outputTensorUser2.host<float>();

        for( int i = 0; i < n_cls; i++ )
        {
            if( cls_data[( i * c_cls + 1 ) * h_cls * w_cls] > RNET_THRESH )
            {
                float sn = reg_data[( i * c_reg + 0 ) * h_reg * w_reg];
                float xn = reg_data[( i * c_reg + 1 ) * h_reg * w_reg];
                float yn = reg_data[( i * c_reg + 2 ) * h_reg * w_reg];

                int rx, ry, rw;
                int crop_x = tmp_winList[i].x;
                int crop_y = tmp_winList[i].y;
                int crop_w = tmp_winList[i].width;
                rw = int( sn * crop_w );
                rx = int( crop_x - 0.5 * sn * crop_w + crop_w * sn * xn + 0.5 * crop_w );
                ry = int( crop_y - 0.5 * sn * crop_w + crop_w * sn * yn + 0.5 * crop_w );

                if( Legal( rx, ry, inputImageWidth, inputImageHeight ) && Legal( rx + rw - 1, ry + rw - 1, inputImageWidth, inputImageHeight ) )
                {
                    ret.push_back( RegRect( rx, ry, rw, rw, tmp_winList[i].scale, cls_data[( i * c_cls + 1 ) * h_cls * w_cls] ) );
                }
            }
        }

    }
    ret = NMS(ret, true, RNET_NMS_THRESH);
    return ret;
}

std::vector<RegRect> MTCNN::runOnet(std::vector<RegRect> &winList)
{
    if( winList.size() == 0 )
        return winList;
    std::vector<RegRect> ret;
    while( winList.size() )
    {
        std::vector<RegRect> tmp_winList;
        int count = winList.size() < onetInputBatch ? winList.size():onetInputBatch;
        while( winList.size() && count )
        {
            tmp_winList.push_back( winList.back() );
            winList.pop_back();
            count--;
        }
        //for test
        tmp_winList[0].x = 181;
        tmp_winList[0].y = 98;
        tmp_winList[1].x = 203;
        tmp_winList[1].y = 108;
        tmp_winList[2].x = 197;
        tmp_winList[2].y = 133;
        tmp_winList[3].x = 215;
        tmp_winList[3].y = 120;
        tmp_winList[4].x = 198;
        tmp_winList[4].y = 107;
        //
        Tensor inputTensorUser1(onetInputTensors[0], onetInputTensors[0]->getDimensionType());
        netInputPreprocess(tmp_winList, inputTensorUser1, onetInputChannel, onetInputWidth, onetInputHeight);
        onetInputTensors[0]->copyFromHostTensor(&inputTensorUser1);
        onet->runSession(onetSession);
        Tensor outputTensorUser1(onetOutputTensors[0], onetOutputTensors[0]->getDimensionType());
        Tensor outputTensorUser2(onetOutputTensors[1], onetOutputTensors[1]->getDimensionType());
        onetOutputTensors[0]->copyToHostTensor(&outputTensorUser1);
        onetOutputTensors[1]->copyToHostTensor(&outputTensorUser2);

        auto cls_shape = outputTensorUser1.shape();
        auto reg_shape = outputTensorUser2.shape();
        int n_reg = reg_shape[0];
        int c_reg = reg_shape[1];
        int h_reg = 1;
        int w_reg = 1;
        int n_cls = cls_shape[0];
        int c_cls = cls_shape[1];
        int h_cls = 1;
        int w_cls = 1;

        float *cls_data = outputTensorUser1.host<float>();
        float *reg_data = outputTensorUser2.host<float>();


        for( int i = 0; i < n_cls; i++ )
        {
            if( cls_data[( i * c_cls + 1 ) * h_cls * w_cls] > ONET_THRESH )
            {
                float sn = reg_data[( i * c_reg + 0 ) * h_reg * w_reg];
                float xn = reg_data[( i * c_reg + 1 ) * h_reg * w_reg];
                float yn = reg_data[( i * c_reg + 2 ) * h_reg * w_reg];

                int rx, ry, rw;

                int crop_x = tmp_winList[i].x;
                int crop_y = tmp_winList[i].y;
                int crop_w = tmp_winList[i].width;
                rw = int( sn * crop_w );
                rx = int( crop_x - 0.5 * sn * crop_w + crop_w * sn * xn + 0.5 * crop_w );
                ry = int( crop_y - 0.5 * sn * crop_w + crop_w * sn * yn + 0.5 * crop_w );


                if( Legal( rx, ry, inputImageWidth, inputImageHeight  ) && Legal( rx + rw - 1, ry + rw - 1, inputImageWidth, inputImageHeight  ) )
                {
                    ret.push_back( RegRect( rx, ry, rw, rw, tmp_winList[i].scale, cls_data[( i * c_cls + 1 ) * h_cls * w_cls] ) );
                }
            }
        }
    }
    ret = NMS(ret, false, ONET_NMS_THRESH);
    return ret;
}

#define  CLAMP(x, l, u)   ((x) < (l) ? (l) : ((x) > (u) ? (u) : (x)))
void MTCNN::transWindowToFace(std::vector<RegRect> &winList)
{
    for( size_t i = 0; i < winList.size(); i++ )
    {
        winList[i].y -= int( 0.1 * winList[i].height );
        winList[i].height = int( 1.2 * winList[i].height );        
        int x1 = CLAMP( winList[i].x, 0, inputImageWidth - 1 );
        int y1 = CLAMP( winList[i].y, 0, inputImageHeight - 1 );
        int x2 = CLAMP( winList[i].x + winList[i].width - 1, 0, inputImageWidth - 1 );
        int y2 = CLAMP( winList[i].y + winList[i].height - 1, 0, inputImageHeight - 1 );
        int w = x2 - x1 + 1;
        int h = y2 - y1 + 1;
        if( w > 0 && h > 0 )
        {
            FaceInfo f;
            f.pos.x = x1;
            f.pos.y = y1;
            f.pos.width = w;
            f.pos.height = h;
            f.score = float(winList[i].conf);
            detectFaceInfo.push_back( f );
        }
    }
}

std::vector<FaceInfo> MTCNN::runFaceDect(ImageData imageInfo)
{
    inputImageData = imageInfo.data;
    inputImageWidth = imageInfo.width;
    inputImageHeight = imageInfo.height;
    inputImageChannel = imageInfo.channels;
    inputImageChannelType = imageInfo.channel_type;

    std::vector<RegRect> winList;
    winList = runPnet();
    winList = runRnet(winList);
    winList = runOnet(winList);

    detectFaceInfo.clear();
    //for test
    winList[0].x = 203;
    winList[0].y = 110;
    winList[0].width = 133;
    winList[0].height = 133;
    //
    transWindowToFace( winList );
    return detectFaceInfo;

}


void MTCNN::shapeIndexPatch(float*input1, std::vector<int> dim1, float*input2, std::vector<int> dim2, float* output)
{
    int feat_h = dim1[2];
    int feat_w = dim1[3];

    int landmarkx2 = dim2[1];
    int x_patch_h = int( originPatch * dim1[2] / float( landmarkNet1InputHeight ) + 0.5f );
    int x_patch_w = int( originPatch * dim1[3] / float( landmarkNet1InputWidth ) + 0.5f );

    int feat_patch_h = x_patch_h;
    int feat_patch_w = x_patch_w;

    int num = dim1[0];
    int channels = dim1[1];

    const float r_h = ( feat_patch_h - 1 ) / 2.0f;
    const float r_w = ( feat_patch_w - 1 ) / 2.0f;
    const int landmark_num = int(landmarkx2 * 0.5);
    HypeShape pos_offset( {dim2[0], dim2[1]} );
    HypeShape feat_offset( {dim1[0], dim1[1], dim1[2], dim1[3]} );
    int nmarks = int( landmarkx2 * 0.5 );
    HypeShape out_offset( {dim1[0], dim1[1], x_patch_h, nmarks, x_patch_w} );

    float *buff = output;
    const float *feat_data = input1;
    const float *pos_data  = input2;
    float zero = 0;

    for( int i = 0; i < landmark_num; i++ )
    {
        for( int n = 0; n < num; n++ )  // x1, y1, ..., xn, yn
        {
            // coordinate of the first patch pixel, scale to the feature map coordinate
            const int y = int( pos_data[pos_offset.to_index( {n, 2 * i + 1} )] * ( feat_h - 1 ) - r_h + 0.5f );
            const int x = int( pos_data[pos_offset.to_index( {n, 2 * i} )] * ( feat_w - 1 ) - r_w + 0.5f );

            for( int c = 0; c < channels; c++ )
            {
                for( int ph = 0; ph < feat_patch_h; ph++ )
                {
                    for( int pw = 0; pw < feat_patch_w; pw++ )
                    {
                        const int y_p = y + ph;
                        const int x_p = x + pw;
                        // set zero if exceed the img bound
                        if( y_p < 0 || y_p >= feat_h || x_p < 0 || x_p >= feat_w )
                            buff[out_offset.to_index( {n, c, ph, i, pw} )] = zero;
                        else
                            buff[out_offset.to_index( {n, c, ph, i, pw} )] =
                                feat_data[feat_offset.to_index( {n, c, y_p, x_p} )];
                    }
                }
            }
        }
    }

}

std::vector<FacePoint> MTCNN::runDetecPoints(FaceInfo faceInfo)
{
    detectFacePoints.clear();

    // bounding box
    double width = faceInfo.pos.width - 1, height = faceInfo.pos.height - 1;
    double min_x = faceInfo.pos.x, max_x = faceInfo.pos.x + width;
    double min_y = faceInfo.pos.y, max_y = faceInfo.pos.y + height;

    //make the bounding box square
    double center_x = ( min_x + max_x ) / 2.0, center_y = ( min_y + max_y ) / 2.0;
    double r = ( ( width > height ) ? width : height ) / 2.0;
    min_x = center_x - r;
    max_x = center_x + r;
    min_y = center_y - r;
    max_y = center_y + r;
    width = max_x - min_x + 1;
    height = max_y - min_y + 1;

    //crop face rect
    RegRect faceWin;
    faceWin.height = height;
    faceWin.width = width;
    faceWin.x     = min_x;
    faceWin.y     = min_y;

    //run session net1
    Tensor inputTensorUser1(landmarkInputTensors1[0], landmarkInputTensors1[0]->getDimensionType());
    //netInputPreprocess(faceWin, inputTensorUser1, landmarkNet1InputChannel, landmarkNet1InputWidth, landmarkNet1InputHeight);
    // std::vector<int> shape = inputTensorUser1.shape();
    float *data = inputTensorUser1.host<float>();

    FILE *f = fopen("/home/xj-zjd/work_space/self_work/seetaface2_code/MNN/build/memorydata.txt", "r");
    for(int i = 0; i < inputTensorUser1.size() / sizeof(float); i++)
    {
        float temp = 0.0f;
        fscanf(f, "%f", &temp);
        data[i] = temp;
    }
    fclose(f);


    // for(int batch = 0; batch < shape[0]; batch++)
    // {
    //     int batchStride = batch * shape[1] * shape[2] * shape[3];
    //     for(int i = 0; i < shape[1]; i++)
    //     {
    //         for(int j = 0; j < shape[2]; j++)
    //         {
    //             for(int k = 0; k < shape[3]; k++)
    //             {
    //                 printf("%f ", data[batchStride + i*shape[2]*shape[3] + j*shape[3] +k]);
    //             }
    //             printf("\n");
    //         }
    //         printf("\n");
    //     }
    //     printf("\n");
    // }    
    
    landmarkInputTensors1[0]->copyFromHostTensor(&inputTensorUser1);
    landmarkNet1->runSession(landmarkNetSession1);
    Tensor outputTensorUser1(landmarkOutputTensors1[0], landmarkOutputTensors1[0]->getDimensionType());
    Tensor outputTensorUser2(landmarkOutputTensors1[1], landmarkOutputTensors1[1]->getDimensionType());
    landmarkOutputTensors1[0]->copyToHostTensor(&outputTensorUser1);
    landmarkOutputTensors1[1]->copyToHostTensor(&outputTensorUser2);

    // Tensor* tempOut = landmarkNet1->getSessionOutput(landmarkNetSession1, "output2");
    // Tensor tempOutUser(tempOut, tempOut->getDimensionType());
    // tempOut->copyToHostTensor(&tempOutUser);
    // // std::vector<int> shape = tempUser.shape();
    // // float *data = tempUser.host<float>();
    // shape = tempOutUser.shape();
    // data = tempOutUser.host<float>();
    // shape.push_back(1);
    // shape.push_back(1);
    // for(int batch = 0; batch < shape[0]; batch++)
    // {
    //     int batchStride = batch * shape[1] * shape[2] * shape[3];
    //     for(int i = 0; i < shape[1]; i++)
    //     {
    //         for(int j = 0; j < shape[2]; j++)
    //         {
    //             for(int k = 0; k < shape[3]; k++)
    //             {
    //                 printf("%f ", data[batchStride + i*shape[2]*shape[3] + j*shape[3] +k]);
    //             }
    //             printf("\n");
    //         }
    //         printf("\n");
    //     }
    //     printf("\n");
    // }

    //shapeindexpatch process
    Tensor inputTensorUser2(landmarkInputTensors2[0], landmarkInputTensors2[0]->getDimensionType());
    float *input1 = outputTensorUser1.host<float>();
    float *input2 = outputTensorUser2.host<float>();
    auto shape1 = outputTensorUser1.shape();
    auto shape2 = outputTensorUser2.shape();
    float *output = inputTensorUser2.host<float>();
    shapeIndexPatch(input1, shape1, input2, shape2, output);


    // shape = inputTensorUser2.shape();
    // data = output;
    // shape[0] = 1;
    // shape[1] = 8;
    // shape[2] = 4;
    // shape[3] = 20;
    // for(int batch = 0; batch < shape[0]; batch++)
    // {
    //     int batchStride = batch * shape[1] * shape[2] * shape[3];
    //     for(int i = 0; i < shape[1]; i++)
    //     {
    //         for(int j = 0; j < shape[2]; j++)
    //         {
    //             for(int k = 0; k < shape[3]; k++)
    //             {
    //                 printf("%f ", data[batchStride + i*shape[2]*shape[3] + j*shape[3] +k]);
    //             }
    //             printf("\n");
    //         }
    //         printf("\n");
    //     }
    //     printf("\n");
    // }

    //run session net2
    landmarkInputTensors2[0]->copyFromHostTensor(&inputTensorUser2);
    landmarkNet2->runSession(landmarkNetSession2);
    Tensor outputTensorUser3(landmarkOutputTensors2[0], landmarkOutputTensors2[0]->getDimensionType());
    landmarkOutputTensors2[0]->copyToHostTensor(&outputTensorUser3);

    //post process
    float *left = outputTensorUser2.host<float>();
    float *right = outputTensorUser3.host<float>();
    int size = outputTensorUser2.size() / sizeof(float);
    for(int i = 0; i < size /2; i++)
    {
        FacePoint pointLoc;
        pointLoc.x = (left[2*i] + right[2*i]) * width + min_x;
        pointLoc.y = (left[2*i+1] + right[2*i+1]) * height + min_y;
        detectFacePoints.push_back(pointLoc);
        printf("%f %f ", pointLoc.x, pointLoc.y);
    }
    printf("\n");
    return detectFacePoints;
}



void MTCNN::cropFace(std::vector<FacePoint> &facePoint, int height, int width, int channels, uint8_t* faceData)
{
    float mean_shape[10] =
    {
        89.3095f, 72.9025f,
        169.3095f, 72.9025f,
        127.8949f, 127.0441f,
        96.8796f, 184.8907f,
        159.1065f, 184.7601f,
    };

    float points[10];
    for( int i = 0; i < 5; ++i )
    {
        points[2 * i] = float( facePoint[i].x );
        points[2 * i + 1] = float( facePoint[i].y );
    }
    std::unique_ptr<uint8_t[]> tempFaceData( new uint8_t[channels * width * height]);
    face_crop_core( inputImageData, inputImageWidth, inputImageHeight, inputImageChannel, tempFaceData.get(), 
    CROP_FACE_WIDTH, CROP_FACE_HEIGHT, points, 5, mean_shape, 256, 256 );

    RegRect win;
    win.width = faceRecogNetInputWidth;
    win.height = faceRecogNetInputHeight;
    win.x    = (width - faceRecogNetInputWidth) / 2;
    win.y    = (height - faceRecogNetInputHeight) / 2;
    cropImage(tempFaceData.get(), width, height, channels, &win, faceData);
}

static void normalize( float *features, int num )
{
    double norm = 0;
    float *dim = features;
    for( int i = 0; i < num; ++i )
    {
        norm += *dim * *dim;
        ++dim;
    }
    norm = std::sqrt( norm ) + 1e-5;
    dim = features;
    for( int i = 0; i < num; ++i )
    {
        *dim /= float( norm );
        ++dim;
    }
}

void MTCNN::ExtractFeature(uint8_t *faceData, int width ,int height)
{
    //run face recog session
    Tensor inputTensorUser(faceRecogInputTensors[0], faceRecogInputTensors[0]->getDimensionType());
    faceImgPreprocess(faceData, width, height, faceRecogNetInputChannel, faceRecogNetInputWidth, faceRecogNetInputHeight, inputTensorUser);
    // std::vector<int> shape = inputTensorUser1.shape();
    faceRecogInputTensors[0]->copyFromHostTensor(&inputTensorUser);
    faceRecogNet->runSession(faceRecogNetSession);
    Tensor outputTensorUser(faceRecogOutputTensors[0], faceRecogOutputTensors[0]->getDimensionType());
    faceRecogOutputTensors[0]->copyToHostTensor(&outputTensorUser);

    ///for test
    // Tensor* tempOut = faceRecogNet->getSessionOutput(faceRecogNetSession, "353");
    // Tensor tempOutUser(tempOut, tempOut->getDimensionType());
    // tempOut->copyToHostTensor(&tempOutUser);
    // auto shape = tempOutUser.shape();
    // auto data = tempOutUser.host<float>();
    // for(int batch = 0; batch < shape[0]; batch++)
    // {
    //     int batchStride = batch * shape[1] * shape[2] * shape[3];
    //     for(int i = 0; i < shape[1]; i++)
    //     {
    //         for(int j = 0; j < shape[2]; j++)
    //         {
    //             for(int k = 0; k < shape[3]; k++)
    //             {
    //                 printf("%f ", data[batchStride + i*shape[2]*shape[3] + j*shape[3] +k]);
    //             }
    //             printf("\n");
    //         }
    //         printf("\n");
    //     }
    //     printf("\n");
    // }

    float *data = outputTensorUser.host<float>();
    int size = outputTensorUser.size() / sizeof(float);
    for(int i = 0; i < size; i++)
    {
        faceFeature.push_back(data[i]);
        // if(i % 32 == 0)
        //     printf("\n");
        // printf("%f ", data[i]);
    }
    // printf("\n");
    normalize(&faceFeature[0], size);
}

std::vector<float> MTCNN::getFaceFeature(std::vector<FacePoint> &facePoint)
{
    int height = CROP_FACE_HEIGHT;
    int width  = CROP_FACE_WIDTH;
    int channels = 3;
    if (inputImageChannelType == Channel_BGRA || inputImageChannelType == Channel_RGBA)
        channels = 4;
    std::unique_ptr<uint8_t[]> faceData( new uint8_t[channels * faceRecogNetInputWidth * faceRecogNetInputHeight]);
    cropFace(facePoint, height, width, channels, faceData.get());
    ExtractFeature(faceData.get(), faceRecogNetInputWidth, faceRecogNetInputHeight);
    return faceFeature;
}
float MTCNN::computeSimilarity(std::vector<float> feature1, std::vector<float> feature2)
{
    if(feature1.size() != feature2.size())
    {
        printf("input feature1 size(%d) and feature size(%d) not match", int(feature1.size()), int(feature2.size()));
        return 0.0f;
    }
    float similarity = 0.0f;
    int size = feature1.size();
    for(int i = 0; i < size; i++)
    {
        similarity += (feature1[i] * feature2[i]);
    }
    return similarity;
}

MTCNN* MTCNN::get_instance()
{
    return mtcnn;
}

MTCNN::MTCNN()
{
    /*********************************************************
     *********************************************************/
    /*detect face net init*/
    printf("init detect face net start!!!!!!!!!!!!!!!!!\n");
    {
        //1------------- init pnet
        pnet.reset(Interpreter::createFromFile(pnetName));
        ScheduleConfig config;
        // config.saveTensors.push_back("14");
        config.saveTensors.push_back("output1");
        config.saveTensors.push_back("output2");
        config.type  = MNN_FORWARD_CPU;
        config.numThread = 4;

        //init net session
        pnetSession = pnet->createSession(config);
        pnetInputTensors.push_back(pnet->getSessionInput(pnetSession, NULL));
        pnetOutputTensors.push_back(pnet->getSessionOutput(pnetSession, "output1"));
        pnetOutputTensors.push_back(pnet->getSessionOutput(pnetSession, "output2"));

        auto shape = pnetInputTensors[0]->shape();
        pnetInputBatch = shape[0];
        pnetInputChannel = shape[1];
        pnetInputHeight = shape[2];
        pnetInputWidth = shape[3];
    }
    {
        //2------------- init rnet
        rnet.reset(Interpreter::createFromFile(rnetName));
        ScheduleConfig config;
        config.saveTensors.push_back("output1");
        config.saveTensors.push_back("output2");
        config.type  = MNN_FORWARD_CPU;
        config.numThread = 4;

        //init net session
        rnetSession = rnet->createSession(config);
        rnetInputTensors.push_back(rnet->getSessionInput(rnetSession, NULL));
        rnetOutputTensors.push_back(rnet->getSessionOutput(rnetSession, "output1"));
        rnetOutputTensors.push_back(rnet->getSessionOutput(rnetSession, "output2"));

        auto shape = rnetInputTensors[0]->shape();
        rnetInputBatch = shape[0];
        rnetInputChannel = shape[1];
        rnetInputHeight = shape[2];
        rnetInputWidth = shape[3];
    }
    {
        //3------------- init onet
        onet.reset(Interpreter::createFromFile(onetName));
        ScheduleConfig config;
        // config.saveTensors.push_back("14");
        config.saveTensors.push_back("output1");
        config.saveTensors.push_back("output2");
        config.type  = MNN_FORWARD_CPU;
        config.numThread = 4;

        //init net session
        onetSession = onet->createSession(config);
        onetInputTensors.push_back(onet->getSessionInput(onetSession, NULL));
        onetOutputTensors.push_back(onet->getSessionOutput(onetSession, "output1"));
        onetOutputTensors.push_back(onet->getSessionOutput(onetSession, "output2"));  

        auto shape = onetInputTensors[0]->shape();
        onetInputBatch = shape[0];
        onetInputChannel = shape[1];
        onetInputHeight = shape[2];
        onetInputWidth = shape[3];
    }
    printf("init detect face net done!!!!!!!!!!!!!!!!!\n");

    /*********************************************************
     *********************************************************/
    //init landmark net 
    printf("init face landmark net start!!!!!!!!!!!!!!!!!\n");
    {
        //1------------- init landmark net1
        landmarkNet1.reset(Interpreter::createFromFile(landmarkNetName1));
        ScheduleConfig config;
        // config.saveTensors.push_back("32");
        config.saveTensors.push_back("output1");
        config.saveTensors.push_back("output2");
        config.type  = MNN_FORWARD_CPU;
        config.numThread = 4;

        //init net session
        landmarkNetSession1 = landmarkNet1->createSession(config);
        landmarkInputTensors1.push_back(landmarkNet1->getSessionInput(landmarkNetSession1, NULL));
        landmarkOutputTensors1.push_back(landmarkNet1->getSessionOutput(landmarkNetSession1, "output1"));
        landmarkOutputTensors1.push_back(landmarkNet1->getSessionOutput(landmarkNetSession1, "output2"));
    }
    {
        //2------------- init landmark net2
        landmarkNet2.reset(Interpreter::createFromFile(landmarkNetName2));
        ScheduleConfig config;
        // config.saveTensors.push_back("30");
        config.saveTensors.push_back("output");
        config.type  = MNN_FORWARD_CPU;
        config.numThread = 4;

        //init net session
        landmarkNetSession2 = landmarkNet2->createSession(config);
        landmarkInputTensors2.push_back(landmarkNet2->getSessionInput(landmarkNetSession2, NULL));
        landmarkOutputTensors2.push_back(landmarkNet2->getSessionOutput(landmarkNetSession2, "output"));
    }
    printf("init face landmark net done!!!!!!!!!!!!!!!!!\n");

    /*********************************************************
     *********************************************************/
    //init face recognization net
    printf("init face recognization net start!!!!!!!!!!!!!!!!!\n");
    {
        //1------------- init face recognization net
        faceRecogNet.reset(Interpreter::createFromFile(faceRecogNetName));
        ScheduleConfig config;
        config.saveTensors.push_back("353");
        config.saveTensors.push_back("output");
        config.type  = MNN_FORWARD_CPU;
        config.numThread = 4;

        //init net session
        faceRecogNetSession = faceRecogNet->createSession(config);
        faceRecogInputTensors.push_back(faceRecogNet->getSessionInput(faceRecogNetSession, NULL));
        faceRecogOutputTensors.push_back(faceRecogNet->getSessionOutput(faceRecogNetSession, "output"));

        auto shape = faceRecogInputTensors[0]->shape();
        faceRecogNetInputBatch = shape[0];
        faceRecogNetInputChannel = shape[1];
        faceRecogNetInputWidth = shape[2];
        faceRecogNetInputHeight = shape[3];
    }    
    printf("init face recognization net done!!!!!!!!!!!!!!!!!\n");
}


MTCNN* MTCNN::mtcnn = new MTCNN();