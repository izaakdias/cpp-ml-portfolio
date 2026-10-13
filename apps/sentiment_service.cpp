#include <httplib.h>

#include "portfolio/cli.hpp"
#include "portfolio/inference.hpp"
#include "portfolio/tokenizer.hpp"
using namespace portfolio;
int main(int argc, char** argv) {
  return guarded([&] {
    Args args(argc, argv);
    args.only({"model", "vocab", "text", "mode", "length", "host", "port", "help"});
    if (args.has("help")) {
      std::cout
          << "--mode tokenize|predict|serve --text TEXT --model ONNX --vocab TXT --length 128\n";
      return;
    }
    WordPieceTokenizer tokenizer(args.get("vocab", "models/bert/vocab.txt"));
    auto length = args.integer("length", 128);
    auto mode = args.get("mode", "predict");
    if (mode == "tokenize") {
      auto tokens = tokenizer.encode(args.get("text"), length);
      print({{"input_ids", tokens.ids}, {"attention_mask", tokens.mask}});
      return;
    }
    InferenceEngine engine(args.get("model", "models/bert/model.onnx"));
    auto predict = [&](const std::string& text) {
      if (text.empty() || text.size() > 16000 ||
          text.find_first_not_of(" \t\r\n") == std::string::npos)
        throw std::invalid_argument("Nonblank text of at most 16000 bytes required");
      auto tokens = tokenizer.encode(text, length);
      auto response = engine.run(
          {{"input_ids", {{"dtype", "int64"}, {"shape", {1, length}}, {"data", tokens.ids}}},
           {"attention_mask",
            {{"dtype", "int64"}, {"shape", {1, length}}, {"data", tokens.mask}}}});
      auto outputs = response.at("outputs");
      if (outputs.size() != 1)
        throw std::runtime_error("Sentiment model requires one logits output");
      auto logits = outputs.begin().value().at("data").get<std::vector<double>>();
      if (logits.size() != 2)
        throw std::runtime_error("Sentiment model requires two classes");
      double maximum = *std::max_element(logits.begin(), logits.end()),
             negative = std::exp(logits[0] - maximum), positive = std::exp(logits[1] - maximum),
             score = positive / (positive + negative);
      return Json{{"sentiment", score >= 0.5 ? "positive" : "negative"},
                  {"positive_probability", score},
                  {"model_sha256", engine.checksum()},
                  {"tokens_used", std::count(tokens.mask.begin(), tokens.mask.end(), 1)}};
    };
    if (mode == "predict")
      print(predict(args.get("text", "This movie was excellent.")));
    else if (mode == "serve") {
      int port = args.integer("port", 8081);
      if (port < 1 || port > 65535)
        throw std::invalid_argument("Invalid port");
      httplib::Server server;
      server.new_task_queue = [] { return new httplib::ThreadPool(2, 2, 16); };
      server.set_payload_max_length(64 * 1024);
      server.Get("/health", [&](const auto&, auto& response) {
        response.set_content(engine.describe().dump(), "application/json");
      });
      server.Post("/sentiment", [&](const auto& request, auto& response) {
        try {
          auto body = Json::parse(request.body);
          if (body.size() != 1 || !body.contains("text") || !body.at("text").is_string())
            throw std::invalid_argument("Request requires exactly text");
          response.set_content(predict(body.at("text")).dump(), "application/json");
        } catch (const std::exception& error) {
          response.status = 400;
          response.set_content(Json({{"error", error.what()}}).dump(), "application/json");
        }
      });
      if (!server.listen(args.get("host", "127.0.0.1"), port))
        throw std::runtime_error("Cannot listen");
    } else
      throw std::invalid_argument("Unknown mode");
  });
}
