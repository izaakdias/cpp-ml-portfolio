#include <torch/torch.h>

#include <fstream>
#include <numeric>

#include "portfolio/cli.hpp"
#include "portfolio/registry.hpp"
using namespace portfolio;
struct LeNet : torch::nn::Module {
  torch::nn::Conv2d conv1{nullptr}, conv2{nullptr};
  torch::nn::Linear fc1{nullptr}, fc2{nullptr}, fc3{nullptr};
  LeNet() {
    conv1 = register_module("conv1", torch::nn::Conv2d(1, 6, 5));
    conv2 = register_module("conv2", torch::nn::Conv2d(6, 16, 5));
    fc1 = register_module("fc1", torch::nn::Linear(256, 120));
    fc2 = register_module("fc2", torch::nn::Linear(120, 84));
    fc3 = register_module("fc3", torch::nn::Linear(84, 10));
  }
  torch::Tensor forward(torch::Tensor x) {
    x = torch::avg_pool2d(torch::relu(conv1(x)), 2);
    x = torch::avg_pool2d(torch::relu(conv2(x)), 2).flatten(1);
    return fc3(torch::relu(fc2(torch::relu(fc1(x)))));
  }
};
static uint32_t integer(std::ifstream& file) {
  unsigned char bytes[4];
  if (!file.read(reinterpret_cast<char*>(bytes), 4))
    throw std::runtime_error("Truncated IDX header");
  return uint32_t(bytes[0]) << 24 | uint32_t(bytes[1]) << 16 | uint32_t(bytes[2]) << 8 | bytes[3];
}
static torch::Tensor idx(const std::filesystem::path& path, bool images) {
  std::ifstream file(path, std::ios::binary);
  if (!file)
    throw std::runtime_error("Missing IDX file: " + path.string());
  auto magic = integer(file), count = integer(file);
  if (magic != (images ? 2051u : 2049u) || count == 0 || count > 100000)
    throw std::invalid_argument("Invalid IDX dimensions/magic");
  if (images && (integer(file) != 28 || integer(file) != 28))
    throw std::invalid_argument("MNIST requires 28x28 images");
  std::vector<uint8_t> bytes(count * (images ? 784 : 1));
  if (!file.read(reinterpret_cast<char*>(bytes.data()), bytes.size()) || file.peek() != EOF)
    throw std::runtime_error("Invalid IDX payload length");
  if (!images && *std::max_element(bytes.begin(), bytes.end()) > 9)
    throw std::invalid_argument("Invalid MNIST class");
  auto tensor =
      torch::from_blob(bytes.data(), {static_cast<int64_t>(bytes.size())}, torch::kUInt8).clone();
  return images ? tensor.reshape({count, 1, 28, 28}).to(torch::kFloat32) / 255.0
                : tensor.to(torch::kInt64);
}
static double accuracy(LeNet& model, const torch::Tensor& x, const torch::Tensor& y) {
  torch::NoGradGuard guard;
  model.eval();
  int64_t correct = 0;
  for (int64_t i = 0; i < x.size(0); i += 256) {
    auto n = std::min(int64_t(256), x.size(0) - i);
    correct +=
        model.forward(x.narrow(0, i, n)).argmax(1).eq(y.narrow(0, i, n)).sum().item<int64_t>();
  }
  return double(correct) / y.size(0);
}
int main(int argc, char** argv) {
  return guarded([&] {
    Args args(argc, argv);
    args.only({"data", "epochs", "limit", "artifacts", "help"});
    if (args.has("help")) {
      std::cout << "--data data/mnist --epochs 3 --limit 12000 --artifacts artifacts\n";
      return;
    }
    int epochs = args.integer("epochs", 3), limit = args.integer("limit", 12000);
    if (epochs < 1 || epochs > 100 || limit < 1000 || limit > 60000)
      throw std::invalid_argument("epochs 1..100 and limit 1000..60000 required");
    torch::manual_seed(41);
    torch::set_num_threads(2);
    auto directory = std::filesystem::path(args.get("data", "data/mnist"));
    auto x = idx(directory / "train-images-idx3-ubyte", true),
         y = idx(directory / "train-labels-idx1-ubyte", false);
    auto tx = idx(directory / "t10k-images-idx3-ubyte", true),
         ty = idx(directory / "t10k-labels-idx1-ubyte", false);
    if (x.size(0) != y.size(0) || tx.size(0) != ty.size(0) || limit > x.size(0))
      throw std::invalid_argument("Inconsistent MNIST sizes");
    auto order = torch::randperm(x.size(0), torch::kInt64).narrow(0, 0, limit);
    x = x.index_select(0, order);
    y = y.index_select(0, order);
    auto n = limit * 9 / 10;
    auto vx = x.narrow(0, n, limit - n), vy = y.narrow(0, n, limit - n);
    x = x.narrow(0, 0, n);
    y = y.narrow(0, 0, n);
    LeNet model;
    torch::optim::Adam optimizer(model.parameters(), torch::optim::AdamOptions(0.001));
    double best = -1;
    Json history = Json::array();
    std::filesystem::path root = args.get("artifacts", "artifacts");
    std::filesystem::create_directories(root / "checkpoints");
    auto checkpoint =
        root / "checkpoints" /
        ("lenet-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()) +
         ".pt");
    for (int epoch = 1; epoch <= epochs; ++epoch) {
      model.train();
      auto shuffled = torch::randperm(n, torch::kInt64);
      double loss_sum = 0;
      for (int i = 0; i < n; i += 64) {
        auto size = std::min(64, n - i);
        auto selected = shuffled.narrow(0, i, size);
        optimizer.zero_grad();
        auto loss = torch::nn::functional::cross_entropy(model.forward(x.index_select(0, selected)),
                                                         y.index_select(0, selected));
        loss.backward();
        optimizer.step();
        loss_sum += loss.item<double>() * size;
      }
      double value = accuracy(model, vx, vy);
      history.push_back(
          {{"epoch", epoch}, {"train_loss", loss_sum / n}, {"validation_accuracy", value}});
      std::cerr << history.back().dump() << '\n';
      if (value > best) {
        best = value;
        torch::serialize::OutputArchive archive;
        model.save(archive);
        archive.save_to(checkpoint.string());
      }
    }
    torch::serialize::InputArchive archive;
    archive.load_from(checkpoint.string());
    model.load(archive);
    model.eval();
    Json parameters = Json::object();
    for (const auto& named : model.named_parameters()) {
      auto tensor = named.value().detach().contiguous();
      parameters[named.key()] = {
          {"shape", tensor.sizes().vec()},
          {"data", std::vector<float>(tensor.data_ptr<float>(),
                                      tensor.data_ptr<float>() + tensor.numel())}};
    }
    Json metrics = {{"test_accuracy", accuracy(model, tx, ty)},
                    {"validation_accuracy", best},
                    {"train_samples", n},
                    {"test_samples", ty.size(0)}};
    torch::NoGradGuard no_grad;
    auto logits = model.forward(tx.narrow(0, 0, 3)).contiguous();
    auto sample_logits =
        std::vector<float>(logits.data_ptr<float>(), logits.data_ptr<float>() + logits.numel());
    auto artifact =
        save_artifact(root, "cnn", {{"architecture", "LeNet"}, {"parameters", parameters}},
                      {{"metrics", metrics},
                       {"sample_logits", sample_logits},
                       {"history", history},
                       {"seed", 41},
                       {"preprocess", "float32 NCHW grayscale /255"},
                       {"train_sha256", sha256(directory / "train-images-idx3-ubyte")}});
    write_json(
        root / "cnn-native-logits.json",
        std::vector<float>(logits.data_ptr<float>(), logits.data_ptr<float>() + logits.numel()));
    print({{"metrics", metrics}, {"artifact", artifact.path.string()}});
  });
}
