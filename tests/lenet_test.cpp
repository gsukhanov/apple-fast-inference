#include <gtest/gtest.h>
#include <opencv2/opencv.hpp>

#include "fastinf/core.hpp"
#include "fastinf/nn.hpp"

using namespace fastinf;
using namespace fastinf::nn;

template <DType _DType, DeviceLikeType _Device = DeviceLikeType::cpu>
class LeNet : public Layer<_DType, _Device> {
 public:
    using tensor_t = Tensor<_DType, _Device>;
    using scalar_t = typename DTypeTraits<_DType>::type;

    LeNet(std::int64_t num_classes = 10)
        : conv1(1, 6, 5),
          relu1(),
          pool1(2, 2),
          conv2(6, 16, 5),
          relu2(),
          pool2(2, 2),
          flatten(),
          fc1(16 * 4 * 4, 120),
          relu3(),
          fc2(120, 84),
          relu4(),
          fc3(84, num_classes) {
        conv1.bind_identity("conv1");
        conv2.bind_identity("conv2");
        fc1.bind_identity("fc1");
        fc2.bind_identity("fc2");
        fc3.bind_identity("fc3");
        this->bind_children(conv1, conv2, fc1, fc2, fc3);
    }

    std::string name() const override {
        return "nn.LeNet";
    }

    tensor_t forward(const tensor_t& x) const override {
        tensor_t out = x;
        out = pool1(relu1(conv1(out)));
        out = pool2(relu2(conv2(out)));

        out = flatten(out);
        out = relu3(fc1(out));
        out = relu4(fc2(out));
        out = fc3(out);

        return out;
    }

 private:
    Conv2d<_DType, _Device> conv1, conv2;
    ReLU<_DType, _Device> relu1, relu2, relu3, relu4;
    AvgPool2d<_DType, _Device> pool1, pool2;
    Flatten<_DType, _Device> flatten;
    Linear<_DType, _Device> fc1, fc2, fc3;
};

int classNum(const std::string& input_file, const std::string& weight) {
    cv::Mat image = cv::imread(input_file, cv::IMREAD_GRAYSCALE);
    if (image.empty()) {
        throw std::runtime_error("Failed to read image");
    }

    cv::resize(image, image, cv::Size(28, 28));
    image.convertTo(image, CV_32F, 1.0 / 255.0);

    LeNet<DType::float32> model;
    Loader<DType::float32> loader;
    model.load_state(loader.load(weight));

    if (!image.isContinuous()) {
        image = image.clone();
    }
    const int sizes[] = {1, 1, image.rows, image.cols};
    image = image.reshape(1, 4, sizes);
    Tensor<DType::float32> input(std::move(image));

    auto out = model(input);
    Softmax<DType::float32> softmax;
    out = softmax(out);
    float max_el = *out.begin();
    int index = 0;
    for (int i = 1; i < 10; ++i) {
        if (max_el < out.at({0, i})) {
            max_el = out.at({0, i});
            index = i;
        }
    }

    return index;
}

TEST(LeNet, test1_9) {
    int index = classNum("tests/inputs_test/test1_9.jpg",
                         "tests/weights/mnist_lenet.json");

    ASSERT_TRUE(index == 9);
}

TEST(LeNet, test2_8) {
    int index = classNum("tests/inputs_test/test2_8.jpg",
                         "tests/weights/mnist_lenet.json");

    ASSERT_TRUE(index == 8);
}

TEST(LeNet, test3_0) {
    int index = classNum("tests/inputs_test/test3_0.jpg",
                         "tests/weights/mnist_lenet.json");

    ASSERT_TRUE(index == 0);
}

TEST(LeNet, test4_2) {
    int index = classNum("tests/inputs_test/test4_2.jpg",
                         "tests/weights/mnist_lenet.json");

    ASSERT_TRUE(index == 2);
}

TEST(LeNet, test5_3) {
    int index = classNum("tests/inputs_test/test5_3.jpg",
                         "tests/weights/mnist_lenet.json");

    ASSERT_TRUE(index == 3);
}

TEST(LeNet, test6_5) {
    int index = classNum("tests/inputs_test/test6_5.jpg",
                         "tests/weights/mnist_lenet.json");

    ASSERT_TRUE(index == 5);
}

TEST(LeNet, test7_6) {
    int index = classNum("tests/inputs_test/test7_6.jpg",
                         "tests/weights/mnist_lenet.json");

    ASSERT_TRUE(index == 6);
}

TEST(LeNet, test8_4) {
    int index = classNum("tests/inputs_test/test8_4.jpg",
                         "tests/weights/mnist_lenet.json");

    ASSERT_TRUE(index == 4);
}

TEST(LeNet, test9_7) {
    int index = classNum("tests/inputs_test/test9_7.jpg",
                         "tests/weights/mnist_lenet.json");

    ASSERT_TRUE(index == 7);
}

TEST(LeNet, test10_1) {
    int index = classNum("tests/inputs_test/test10_1.jpg",
                         "tests/weights/mnist_lenet.json");

    ASSERT_TRUE(index == 1);
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);

    return RUN_ALL_TESTS();
}
