/**
* Copyright 2020 Huawei Technologies Co., Ltd
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at

* http://www.apache.org/licenses/LICENSE-2.0

* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.

* File utils.cpp
* Description: handle file operations
*/
#include "utils.h"
#include <bits/stdint-uintn.h>
#include <cstdint>
#include <map>
#include <iostream>
#include <fstream>
#include <cmath>
#include <unistd.h>
#include <cstring>
#include <dirent.h>
#include <vector>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>
#include "acl/acl.h"

using namespace std;

aclrtRunMode Utils::runMode_ = ACL_DEVICE;
const static std::vector<std::string> Label = {"airplane", "ship", "oiltank", "playground", "port", "bridge", "car"};
const static uint32_t kGridSize[3][2] = {{40, 80}, {20, 40}, {10, 20}};
const uint numBBoxes = 3;
const uint numClasses = 7;
const uint BoxTensorLabel = 12;
const float nmsThresh = 0.45;
const float MaxBoxClassThresh = 0.3;
const int anchor[3][3][2] = {{{10,13},{16,30},{33,23}}, {{30,61},{62,45},{59,119}}, {{116,90},{156,198},{373,326}}};
const int stride[3] = {8, 16, 32};

const uint32_t kLineSolid = 2;
const double kFountScale = 0.5;
const cv::Scalar kFontColor(0,0,255);
const uint32_t kLabelOffset = 11;
const vector<cv::Scalar> kColors{
    cv::Scalar(237, 149, 100), cv::Scalar(0, 215, 255),
    cv::Scalar(50, 205, 50), cv::Scalar(139, 85, 26)};

std::vector<BBox> Utils::PostProcess(const string &path, aclmdlDataset *modelOutput, aclmdlDesc* modelDesc)
{
    vector<BBox> binfo, bboxesNew;
    size_t outDatasetNum = aclmdlGetDatasetNumBuffers(modelOutput);
    if (outDatasetNum != 3) {
        ERROR_LOG("outDatasetNum=%zu must be 3",outDatasetNum);
        return binfo;
    }
    int H,W;
    cv::Mat Image = cv::imread(path, CV_LOAD_IMAGE_COLOR);
    H = Image.rows;
    W = Image.cols;
    float widthScale = float(MODEL_INPUT_WIDTH) / float(W);
    float heightScale = float(MODEL_INPUT_HEIGHT) / float(H);
    //cout<<widthScale<<" "<<heightScale<<endl;
    float x, y, w, h, cf, tx, ty, tw,th;//anchors = [[10, 13, 16, 30, 33, 23], [30, 61, 62, 45, 59, 119], [116, 90, 156, 198, 373, 326]]
    for (size_t i = 0; i < outDatasetNum; i++) {
        const uint32_t gridrow = kGridSize[i][0];
        const uint32_t gridcol = kGridSize[i][1];
        //cout<<gridrow<<" "<<gridcol<<endl;
        uint32_t dataSize = 0;
        float* detectData = (float*)GetInferenceOutputItem(dataSize, modelOutput, i);
        //cout<<detectData[0]<<endl;
        for(uint cx = 0; cx < gridrow; ++cx){
            for(uint cy = 0; cy < gridcol; ++cy){
                for(uint k = 0; k < numBBoxes; ++k){
                    tx = sigmoid(detectData[((k*BoxTensorLabel)*gridrow+cx)*gridcol+cy]);
                    ty = sigmoid(detectData[((k*BoxTensorLabel+1)*gridrow+cx)*gridcol+cy]);
                    tw = sigmoid(detectData[((k*BoxTensorLabel+2)*gridrow+cx)*gridcol+cy]);
                    th = sigmoid(detectData[((k*BoxTensorLabel+3)*gridrow+cx)*gridcol+cy]);
                    cf = sigmoid(detectData[((k*BoxTensorLabel+4)*gridrow+cx)*gridcol+cy]);
                    //if(cx==0 and cy==0) cout<<tx<<" "<<ty<<" "<<tw<<" "<<th<<" "<<cf<<endl;
                    //find best precision
                    float Maxclass = 0.0f;
                    uint32_t Maxclass_Loc = -1;
                    for(int j=5; j<BoxTensorLabel; ++j){
                        float class_prob = sigmoid(detectData[((k*BoxTensorLabel+j)*gridrow+cx)*gridcol+cy]);
                        if(Maxclass < class_prob){
                            Maxclass = class_prob;
                            Maxclass_Loc = j-5;
                        }
                    }
                    if(Maxclass_Loc!=-1 and cf * Maxclass >= MaxBoxClassThresh){
                        BBox boundBox;
                        x = (tx*2-0.5+cy)*stride[i];
                        y = (ty*2-0.5+cx)*stride[i];
                        w = (tw*2) * (tw*2) * anchor[i][k][0];
                        h = (th*2) * (th*2) * anchor[i][k][1];
                        //cout<<i<<" "<<cx<<" "<<cy<<" "<<k<<" "<<x<<" "<<y<<" "<<w<<" "<<h<<" "<<tx<<" "<<ty<<" "<<tw<<" "<<th<<" "<<stride[i]<<" "<<anchor[i][k][0]<<endl;
                        boundBox.rect.ltX = (uint32_t)max(((x - w/2.0) / widthScale), 0.0);
                        boundBox.rect.ltY = (uint32_t)max(((y - h/2.0) / heightScale), 0.0);
                        boundBox.rect.rbX = (uint32_t)min(((x + w/2.0) / widthScale), W*1.0);
                        boundBox.rect.rbY = (uint32_t)min(((y + h/2.0) / heightScale), H*1.0);
                        boundBox.score = cf * Maxclass;
                        boundBox.cls = Maxclass_Loc;
                        binfo.push_back(boundBox);
                        //cout<<boundBox.rect.ltX<<" "<<boundBox.rect.ltY<<" "<<boundBox.rect.rbX<<" "<<boundBox.rect.rbY<<endl;
                    }
                }
            }
        }
        if (runMode_ == ACL_HOST) {
            delete[] detectData;
        }
    }
    //NMS
    bboxesNew = Utils::nmsAllClasses(nmsThresh, binfo, numClasses);
    return bboxesNew;
}

void* Utils::GetInferenceOutputItem(uint32_t& itemDataSize, aclmdlDataset* inferenceOutput, uint32_t idx){
    aclDataBuffer* dataBuffer = aclmdlGetDatasetBuffer(inferenceOutput, idx);
    if (dataBuffer == nullptr) {
        ERROR_LOG("Get the %dth dataset buffer from model "
        "inference output failed", idx);
        return nullptr;
    }

    void* dataBufferDev = aclGetDataBufferAddr(dataBuffer);
    if (dataBufferDev == nullptr) {
        ERROR_LOG("Get the %dth dataset buffer address "
        "from model inference output failed", idx);
        return nullptr;
    }

    size_t bufferSize = aclGetDataBufferSize(dataBuffer);
    if (bufferSize == 0) {
        ERROR_LOG("The %dth dataset buffer size of "
        "model inference output is 0", idx);
        return nullptr;
    }

    void* data = nullptr;
    if (runMode_ == ACL_HOST) {
        data = Utils::CopyDataDeviceToLocal(dataBufferDev, bufferSize);
        if (data == nullptr) {
            ERROR_LOG("Copy inference output to host failed");
            return nullptr;
        }
    }
    else {
        data = dataBufferDev;
    }
    itemDataSize = bufferSize; //size
    return data; //address
}


bool Utils::IsDirectory(const string &path) {
  // get path stat
  struct stat buf;
  if (stat(path.c_str(), &buf) != kStatSuccess) {
    return false;
  }

  // check
  if (S_ISDIR(buf.st_mode)) {
    return true;
  } else {
    return false;
  }
}

bool Utils::IsPathExist(const string &path) {
  ifstream file(path);
  if (!file) {
    return false;
  }
  return true;
}

void Utils::SplitPath(const string &path, vector<string> &path_vec) {
    char *char_path = const_cast<char*>(path.c_str());
    const char *char_split = kImagePathSeparator.c_str();
    char *tmp_path = strtok(char_path, char_split);
    while (tmp_path) {
        path_vec.emplace_back(tmp_path);
        tmp_path = strtok(nullptr, char_split);
    }
}

void Utils::GetAllFiles(const string &path, vector<string> &file_vec) {
    // split file path
    vector<string> path_vector;
    SplitPath(path, path_vector);

    for (string every_path : path_vector) {
        // check path exist or not
        if (!IsPathExist(path)) {
        ERROR_LOG("Failed to deal path=%s. Reason: not exist or can not access.",
                every_path.c_str());
        continue;
        }
        // get files in path and sub-path
        GetPathFiles(every_path, file_vec);
    }
}

void Utils::GetPathFiles(const string &path, vector<string> &file_vec) {
    struct dirent *dirent_ptr = nullptr;
    DIR *dir = nullptr;
    if (IsDirectory(path)) {
        dir = opendir(path.c_str());
        while ((dirent_ptr = readdir(dir)) != nullptr) {
            // skip . and ..
            if (dirent_ptr->d_name[0] == '.') {
            continue;
            }

            // file path
            string full_path = path + kPathSeparator + dirent_ptr->d_name;
            // directory need recursion
            if (IsDirectory(full_path)) {
                GetPathFiles(full_path, file_vec);
            } else {
                // put file
                file_vec.emplace_back(full_path);
            }
        }
    } 
    else {
        file_vec.emplace_back(path);
    }
}

void* Utils::CopyDataDeviceToLocal(void* deviceData, uint32_t dataSize) {
    uint8_t* buffer = new uint8_t[dataSize];
    if (buffer == nullptr) {
        ERROR_LOG("New malloc memory failed");
        return nullptr;
    }

    aclError aclRet = aclrtMemcpy(buffer, dataSize, deviceData, dataSize, ACL_MEMCPY_DEVICE_TO_HOST);
    if (aclRet != ACL_ERROR_NONE) {
        ERROR_LOG("Copy device data to local failed, aclRet is %d", aclRet);
        delete[](buffer);
        return nullptr;
    }

    return (void*)buffer;
}

void* Utils::CopyDataToDevice(void* data, uint32_t dataSize, aclrtMemcpyKind policy) {
    void* buffer = nullptr;
    aclError aclRet = aclrtMalloc(&buffer, dataSize, ACL_MEM_MALLOC_HUGE_FIRST);
    if (aclRet != ACL_ERROR_NONE) {
        ERROR_LOG("malloc device data buffer failed, aclRet is %d", aclRet);
        return nullptr;
    }

    aclRet = aclrtMemcpy(buffer, dataSize, data, dataSize, policy);
    if (aclRet != ACL_ERROR_NONE) {
        ERROR_LOG("Copy data to device failed, aclRet is %d", aclRet);
        (void)aclrtFree(buffer);
        return nullptr;
    }

    return buffer;
}

void* Utils::CopyDataDeviceToDevice(void* deviceData, uint32_t dataSize) {
    return CopyDataToDevice(deviceData, dataSize, ACL_MEMCPY_DEVICE_TO_DEVICE);
}

void* Utils::CopyDataHostToDevice(void* deviceData, uint32_t dataSize) {
    return CopyDataToDevice(deviceData, dataSize, ACL_MEMCPY_HOST_TO_DEVICE);
}


void Utils::ImageNchw(shared_ptr<ImageDesc>& imageData, std::vector<cv::Mat>& nhwcImageChs, uint32_t size) {
    uint8_t* nchwBuf = new uint8_t[size];
    int channelSize = IMAGE_CHAN_SIZE_F32(nhwcImageChs[0].rows, nhwcImageChs[0].cols);
    int pos = 0;
    for (int i = 0; i < nhwcImageChs.size(); i++) {
        memcpy(static_cast<uint8_t *>(nchwBuf) + pos,  nhwcImageChs[i].ptr<float>(0), channelSize);
        pos += channelSize;
    }

    imageData->size = size;
    imageData->data.reset((uint8_t *)nchwBuf, [](uint8_t* p) { delete[](p);} );
}


bool Utils::PreProcess(shared_ptr<ImageDesc>& imageData, const string& imageFile) {
    //TODO:
    //Read image using opencv
    cv::Mat image = cv::imread(imageFile, CV_LOAD_IMAGE_COLOR);
    if (image.empty()) {
        ERROR_LOG("Read image %s failed", imageFile.c_str());
        return false;
    }
    //resize image to model size
    cv::Mat reiszedImage, rsImageF32, merge_image;
    cv::resize(image, reiszedImage, cv::Size(MODEL_INPUT_WIDTH, MODEL_INPUT_HEIGHT));
    reiszedImage.convertTo(rsImageF32, CV_32FC3);

    //B G R channel means channels[0] channels[1] channels[2]
    std::vector<cv::Mat> channels;
    cv::split(rsImageF32, channels);
    channels[0] = channels[0] / 255.0;
    channels[1] = channels[1] / 255.0;
    channels[2] = channels[2] / 255.0;

    //focus -> split every channel to 4 slice
    std::vector<cv::Mat> B_slice = slice(channels, 0);
    std::vector<cv::Mat> G_slice = slice(channels, 1);
    std::vector<cv::Mat> R_slice = slice(channels, 2);

    std::vector<cv::Mat> merge_channels;
    for(int i=0; i<4; ++i){
        merge_channels.push_back(R_slice[i]);
        merge_channels.push_back(G_slice[i]);
        merge_channels.push_back(B_slice[i]);
    }
    /*
    //cv::merge(merge_channels, merge_image);
    cout<<merge_channels[0].at<float>(50,50)<<endl;
    cout<<merge_channels[1].at<float>(50,50)<<endl;
    cout<<merge_channels[2].at<float>(50,50)<<endl;
    cout<<merge_channels[3].at<float>(50,50)<<endl;
    cout<<merge_channels[4].at<float>(50,50)<<endl;
    cout<<merge_channels[5].at<float>(50,50)<<endl;
    cout<<merge_channels[6].at<float>(50,50)<<endl;
    cout<<merge_channels[7].at<float>(50,50)<<endl;
    cout<<merge_channels[8].at<float>(50,50)<<endl;
    cout<<merge_channels[9].at<float>(50,50)<<endl;
    cout<<merge_channels[10].at<float>(50,50)<<endl;
    cout<<merge_channels[11].at<float>(50,50)<<endl;
    */
    uint32_t size = RGB_IMAGE_SIZE_F32(MODEL_INPUT_WIDTH, MODEL_INPUT_HEIGHT, 12);
    ImageNchw(imageData, merge_channels, size);

    aclError ret = aclrtGetRunMode(&runMode_);
    if (ret != ACL_ERROR_NONE) {
        ERROR_LOG("acl get run mode failed");
    }

    void* imageDev;
    //copy image data to device
    if (runMode_ == ACL_HOST) {
        imageDev = CopyDataHostToDevice(imageData->data.get(), size);
        if (imageDev == nullptr) {
            ERROR_LOG("Copy image info to device failed");
            return FAILED;
        }
    }
    else {
        imageDev = CopyDataDeviceToDevice(imageData->data.get(), size);
        if (imageDev == nullptr) {
            ERROR_LOG("Copy image info to device failed");
            return FAILED;
        }
    }
    imageData->size = size;
    imageData->data.reset((uint8_t *)imageDev, [](uint8_t* p) { aclrtFree(p);});
    return true;
}

std::vector<cv::Mat> Utils::slice(std::vector<cv::Mat>& channels, uint8_t ind){
    std::vector<cv::Mat> temp;
    int dir[4][2] = {{0,0}, {1,0}, {0,1}, {1,1}};
    for(int k=0; k<4; ++k){
        int start_row = dir[k][0];
        int start_col = dir[k][1];
        cv::Mat a(MODEL_INPUT_HEIGHT/2, MODEL_INPUT_WIDTH/2, CV_32FC1);
        for(int i=0; i<MODEL_INPUT_HEIGHT/2; i++){
            for(int j=0; j<MODEL_INPUT_WIDTH/2; j++){
                a.at<float>(i, j) = channels[ind].at<float>(2*i+start_row, 2*j+start_col);
            }
        }
        /*
        for(int i=start_row; i<channels[ind].rows; i=i+2){
            for(int j=start_col; j<channels[ind].cols; j=j+2){
                a.at<float>(i/2, j/2) = channels[ind].at<float>(i, j);
            }
        }
        */
        temp.push_back(a);
    }
    return temp;
}

float Utils::sigmoid(float val){
    return 1.0/(1.0+exp(-val));
}

vector<BBox> Utils::nmsAllClasses(const float nmsThresh, std::vector<BBox>& binfo, const uint numClasses)
{
    std::vector<BBox> result;
    std::vector<std::vector<BBox>> splitBoxes(numClasses);
    for (auto& box : binfo)
    {
        splitBoxes.at(box.cls).push_back(box);
    }

    for (auto& boxes : splitBoxes)
    {
        boxes = nonMaximumSuppression(nmsThresh, boxes);
        result.insert(result.end(), boxes.begin(), boxes.end());
    }

    return result;
}

vector<BBox> Utils::nonMaximumSuppression(const float nmsThresh, std::vector<BBox> binfo)
{
    auto overlap1D = [](float x1min, float x1max, float x2min, float x2max) -> float {
        float left = max(x1min, x2min);
        float right = min(x1max, x2max);
        return right-left;
    };
    auto computeIoU =[&overlap1D](BBox& bbox1, BBox& bbox2) -> float {
    float overlapX = overlap1D(bbox1.rect.ltX, bbox1.rect.rbX, bbox2.rect.ltX, bbox2.rect.rbX);
    float overlapY = overlap1D(bbox1.rect.ltY, bbox1.rect.rbY, bbox2.rect.ltY, bbox2.rect.rbY);
    if(overlapX <= 0 or overlapY <= 0) return 0;
    float area1 = (bbox1.rect.rbX - bbox1.rect.ltX) * (bbox1.rect.rbY - bbox1.rect.ltY);
    float area2 = (bbox2.rect.rbX - bbox2.rect.ltX) * (bbox2.rect.rbY - bbox2.rect.ltY);
    float overlap2D = overlapX * overlapY;
    float u = area1 + area2 - overlap2D;
    return u == 0 ? 0 : overlap2D / u;
};

    std::stable_sort(binfo.begin(), binfo.end(),
    [](const BBox& b1, const BBox& b2) { return b1.score > b2.score;});
    std::vector<BBox> out;
    /*
    //对于每一个检测框 找出iou大于阈值的框 放入invalid
    unordered_set<int> invalid;
    for(int i=0; i<binfo.size(); ++i){
        if(invalid.find(i) != invalid.end()) continue;
        BBox truth = binfo[i];
        for(int j=i+1; j<binfo.size(); ++j){
            if(invalid.find(j) != invalid.end()) continue;
            BBox cur = binfo[j];
            float overlap = computeIoU(cur, truth);
            if(overlap >= nmsThresh) invalid.insert(j);
        }
    }

    for(int i=0; i<binfo.size(); ++i){
        if(invalid.find(i) == invalid.end()) out.push_back(binfo[i]);
    }
    */

    for (auto& i : binfo)
    {
        bool keep = true;
        for (auto& j : out)
        {
            if (keep)
            {
                float overlap = computeIoU(i, j);
                keep = overlap <= nmsThresh;
            }
            else
                break;
        }
        if (keep) out.push_back(i);
    }
    return out;
}

void Utils::DrawBoundBoxToImage(vector<BBox>& detectionResults, const string& origImagePath) {
    cv::Mat image = cv::imread(origImagePath, cv::IMREAD_UNCHANGED);
    for (int i = 0; i < detectionResults.size(); ++i) {
        cv::Point p1, p2;
        p1.x = detectionResults[i].rect.ltX;
        p1.y = detectionResults[i].rect.ltY;
        p2.x = detectionResults[i].rect.rbX;
        p2.y = detectionResults[i].rect.rbY;
        cv::rectangle(image, p1, p2, kColors[i % kColors.size()], kLineSolid);
        string text = Label[detectionResults[i].cls];
        cv::putText(image, text, cv::Point(p1.x, p1.y + kLabelOffset),
        cv::FONT_HERSHEY_COMPLEX, kFountScale, kFontColor);
    }

    int pos = origImagePath.find_last_of("/");
    string filename(origImagePath.substr(pos + 1));
    stringstream sstream;
    sstream.str("");
    sstream << "./output/out_" << filename;
    cv::imwrite(sstream.str(), image);
}

void Utils::WriteBoundBoxToTXT(vector<BBox>& detectionResults, const string& origImagePath) {
    int pos = origImagePath.find_last_of("/");
    string filename(origImagePath.substr(pos + 1));
    pos = filename.find_last_of(".");
    filename = filename.substr(0, pos);
    ofstream out;
    out.open(filename + ".txt", ofstream::app);
    for (int i = 0; i < detectionResults.size(); ++i) {
        int x1, y1, x2, y2;
        x1 = detectionResults[i].rect.ltX;
        y1 = detectionResults[i].rect.ltY;
        x2 = detectionResults[i].rect.rbX;
        y2 = detectionResults[i].rect.rbY;
        string text = Label[detectionResults[i].cls];
        float score = detectionResults[i].score;
        out<<text<<" "<<score<<" "<<x1<<" "<<y1<<" "<<x2<<" "<<y2<<endl;
    }
    out.close();
}