#include <opencv2/opencv.hpp>

#include "portfolio/cli.hpp"
#include "portfolio/inference.hpp"
#include "portfolio/registry.hpp"
using namespace portfolio;
static Json tensor(const cv::Mat& image) {
  if (image.empty())
    throw std::invalid_argument("Empty image");
  cv::Mat resized, rgb;
  cv::resize(image, resized, {224, 224}, 0, 0, cv::INTER_LINEAR);
  cv::cvtColor(resized, rgb, cv::COLOR_BGR2RGB);
  std::vector<float> values(3 * 224 * 224);
  const float mean[] = {.485f, .456f, .406f}, scale[] = {.229f, .224f, .225f};
  for (int y = 0; y < 224; ++y)
    for (int x = 0; x < 224; ++x)
      for (int c = 0; c < 3; ++c)
        values[c * 224 * 224 + y * 224 + x] =
            (rgb.at<cv::Vec3b>(y, x)[c] / 255.0f - mean[c]) / scale[c];
  return {{"image", {{"dtype", "float32"}, {"shape", {1, 3, 224, 224}}, {"data", values}}}};
}
int main(int argc, char** argv) {
  return guarded([&] {
    Args args(argc, argv);
    args.only(
        {"model", "config", "image", "video", "camera", "output", "headless", "frames", "help"});
    if (args.has("help")) {
      std::cout << "--image PATH | --video PATH | --camera 0 --output inspection.jpg --headless "
                   "--frames 100\n";
      return;
    }
    int sources = args.has("image") + args.has("video") + args.has("camera");
    if (sources != 1)
      throw std::invalid_argument("Choose one image, video, or camera source");
    auto config = read_json(args.get("config", "models/vision/config.json"));
    double threshold = config.at("threshold");
    InferenceEngine engine(args.get("model", "models/vision/model.onnx"));
    if (config.at("sha256") != engine.checksum())
      throw std::runtime_error("Vision model checksum mismatch");
    auto inspect = [&](cv::Mat& frame) {
      auto result = engine.run(tensor(frame));
      double score = result.at("outputs").at("score").at("data").at(0);
      bool alert = score >= threshold;
      cv::putText(frame, (alert ? "DEFECT " : "NORMAL ") + std::to_string(score), {15, 35},
                  cv::FONT_HERSHEY_SIMPLEX, .8,
                  alert ? cv::Scalar(0, 0, 255) : cv::Scalar(0, 180, 0), 2);
      return Json{{"score", score},
                  {"threshold", threshold},
                  {"defect", alert},
                  {"model_sha256", engine.checksum()}};
    };
    if (args.has("image")) {
      auto frame = cv::imread(args.get("image"));
      auto result = inspect(frame);
      if (!cv::imwrite(args.get("output", "inspection.jpg"), frame))
        throw std::runtime_error("Cannot save image");
      print(result);
      return;
    }
    cv::VideoCapture capture;
    if (args.has("camera"))
      capture.open(args.integer("camera", 0));
    else
      capture.open(args.get("video"));
    if (!capture.isOpened())
      throw std::runtime_error("Cannot open capture device/video");
    int maximum = args.integer("frames", 300);
    if (maximum < 1 || maximum > 100000)
      throw std::invalid_argument("frames 1..100000 required");
    int count = 0, alerts = 0;
    cv::Mat frame;
    auto start = std::chrono::steady_clock::now();
    while (count < maximum && capture.read(frame)) {
      auto result = inspect(frame);
      alerts += result.at("defect").get<bool>();
      ++count;
      if (!args.has("headless")) {
        cv::imshow("Inspection", frame);
        if (cv::waitKey(1) == 27)
          break;
      }
    }
    if (count == 0)
      throw std::runtime_error("No frames captured");
    if (args.has("output") && !cv::imwrite(args.get("output"), frame))
      throw std::runtime_error("Cannot save final frame");
    double seconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    print({{"frames", count},
           {"alerts", alerts},
           {"wall_seconds", seconds},
           {"fps", count / seconds}});
  });
}
