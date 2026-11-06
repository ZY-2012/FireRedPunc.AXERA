#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "ax_engine_api.h"
#include "ax_sys_api.h"
#include "src/bert_tokenizer.hpp"
#include "src/engine_wrapper.hpp"
#include "src/punc_text.hpp"

#ifndef DEFAULT_MODELS_DIR
#define DEFAULT_MODELS_DIR "models"
#endif

namespace {
using fireredpunc::AddPunc;
using fireredpunc::HasSentPunct;
using fireredpunc::RuleFix;
using fireredpunc::StripMarks;
using fireredpunc::Trim;

constexpr int kSeqLen = 128;
constexpr int kMaxTokens = kSeqLen - 1;
constexpr int kClsId = 101;
constexpr int kLabels = 5;
using Clock = std::chrono::steady_clock;

std::string JoinPath(const std::string& a, const std::string& b) {
  if (a.empty() || a.back() == '/') return a + b;
  return a + "/" + b;
}

class AxRuntime {
 public:
  AxRuntime() {
    if (AX_SYS_Init() != 0) throw std::runtime_error("AX_SYS_Init failed");
    sys_initialized_ = true;
    AX_ENGINE_NPU_ATTR_T attr{};
    if (AX_ENGINE_Init(&attr) != 0) {
      AX_SYS_Deinit();
      sys_initialized_ = false;
      throw std::runtime_error("AX_ENGINE_Init failed");
    }
    engine_initialized_ = true;
  }
  ~AxRuntime() {
    if (engine_initialized_) AX_ENGINE_Deinit();
    if (sys_initialized_) AX_SYS_Deinit();
  }

 private:
  bool sys_initialized_ = false;
  bool engine_initialized_ = false;
};

class PuncModel {
 public:
  explicit PuncModel(std::string model_dir) : model_dir_(std::move(model_dir)) {
    if (!tokenizer_.Load(JoinPath(model_dir_, "vocab.txt"))) {
      throw std::runtime_error("cannot load vocab.txt from " + model_dir_);
    }
    if (!out_dict_.Load(JoinPath(model_dir_, "out_dict"))) {
      throw std::runtime_error("cannot load out_dict from " + model_dir_);
    }
    if (engine_.Init(JoinPath(model_dir_, "punc.axmodel")) != 0) {
      throw std::runtime_error("cannot load punc.axmodel from " + model_dir_);
    }
    const int ids_bytes = engine_.GetInputSizeByName("input_ids");
    const int mask_bytes = engine_.GetInputSizeByName("attention_mask");
    const int logits_bytes = engine_.GetOutputSizeByName("logits");
    if (ids_bytes != kSeqLen * static_cast<int>(sizeof(int32_t)) ||
        mask_bytes != kSeqLen * static_cast<int>(sizeof(float)) ||
        logits_bytes != kMaxTokens * kLabels *
                            static_cast<int>(sizeof(float))) {
      throw std::runtime_error(
          "unexpected model IO: input_ids=" + std::to_string(ids_bytes) +
          " attention_mask=" + std::to_string(mask_bytes) +
          " logits=" + std::to_string(logits_bytes));
    }
  }

  struct Result {
    std::vector<std::string> tokens;
    std::vector<int> ids;
    std::vector<int> preds;
  };

  Result Classify(const std::string& text) {
    const std::string stripped = StripMarks(text);
    Result result;
    result.tokens = tokenizer_.Tokenize(stripped, true);
    result.ids = tokenizer_.ToIds(result.tokens);
    size_t n = std::min(result.ids.size(), static_cast<size_t>(kMaxTokens));
    result.tokens.resize(n);
    result.ids.resize(n);
    if (n == 0) return result;

    std::vector<int32_t> input_ids(kSeqLen, 0);
    std::vector<float> attention_mask(kSeqLen, 0.0f);
    input_ids[0] = kClsId;
    attention_mask[0] = 1.0f;
    for (size_t i = 0; i < n; ++i) {
      input_ids[i + 1] = result.ids[i];
      attention_mask[i + 1] = 1.0f;
    }

    if (engine_.SetInputByName("input_ids", input_ids.data(),
                               input_ids.size() * sizeof(int32_t)) != 0 ||
        engine_.SetInputByName("attention_mask", attention_mask.data(),
                               attention_mask.size() * sizeof(float)) != 0 ||
        engine_.RunSync() != 0) {
      throw std::runtime_error("punc inference failed");
    }

    std::vector<float> logits(static_cast<size_t>(kMaxTokens) * kLabels);
    if (engine_.GetOutputByName("logits", logits.data(),
                                logits.size() * sizeof(float)) != 0) {
      throw std::runtime_error("failed to read logits");
    }
    result.preds.resize(n);
    for (size_t i = 0; i < n; ++i) {
      int best = 0;
      float best_value = logits[i * kLabels];
      for (int k = 1; k < kLabels; ++k) {
        const float value = logits[i * kLabels + k];
        if (value > best_value) {
          best_value = value;
          best = k;
        }
      }
      result.preds[i] = best;
    }
    return result;
  }

  std::string Punctuate(const std::string& input, bool force = false) {
    const std::string text = Trim(input);
    if (text.empty()) return "";
    if (!force && HasSentPunct(text)) return text;
    const Result result = Classify(text);
    if (result.tokens.empty()) return text;
    return RuleFix(AddPunc(result.tokens, result.preds, out_dict_));
  }

  struct Latency {
    double median_ms = 0;
    double p95_ms = 0;
    double min_ms = 0;
    double max_ms = 0;
  };

  Latency Measure(const std::string& text, int runs = 30) {
    Punctuate(text, true);
    std::vector<double> values;
    values.reserve(runs);
    for (int i = 0; i < runs; ++i) {
      const auto t0 = Clock::now();
      Punctuate(text, true);
      const auto t1 = Clock::now();
      values.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
    }
    std::sort(values.begin(), values.end());
    Latency out;
    out.min_ms = values.front();
    out.max_ms = values.back();
    out.median_ms = values[values.size() / 2];
    out.p95_ms = values[static_cast<size_t>(values.size() * 0.95) - 1];
    return out;
  }

 private:
  std::string model_dir_;
  fireredpunc::BertTokenizer tokenizer_;
  fireredpunc::TokenDict out_dict_;
  EngineWrapper engine_;
};

struct Args {
  std::string model_dir = DEFAULT_MODELS_DIR;
  std::vector<std::string> texts;
  std::string text_file;
  bool force = false;
  bool stats = false;
};

void Usage(const char* program) {
  std::printf(
      "Usage: %s --model-dir DIR [--text TEXT]... [--text-file FILE]\n"
      "          [--force] [--stats]\n"
      "  --model-dir   directory containing punc.axmodel, vocab.txt,\n"
      "                out_dict and tokenizer/\n"
      "  --text        text to punctuate; repeatable\n"
      "  --text-file   UTF-8 text file, one sentence per line\n"
      "  --force       run the model even if the input already has .!?。！？\n"
      "  --stats       measure 30-run latency after each output\n",
      program);
}

Args ParseArgs(int argc, char** argv) {
  Args args;
  for (int i = 1; i < argc; ++i) {
    const std::string key = argv[i];
    auto value = [&]() -> std::string {
      if (++i >= argc) throw std::runtime_error("missing value for " + key);
      return argv[i];
    };
    if (key == "--model-dir") {
      args.model_dir = value();
    } else if (key == "--text") {
      args.texts.push_back(value());
    } else if (key == "--text-file") {
      args.text_file = value();
    } else if (key == "--force") {
      args.force = true;
    } else if (key == "--stats") {
      args.stats = true;
    } else if (key == "-h" || key == "--help") {
      Usage(argv[0]);
      std::exit(0);
    } else {
      throw std::runtime_error("unknown argument: " + key);
    }
  }
  if (!args.text_file.empty()) {
    std::ifstream in(args.text_file);
    if (!in) throw std::runtime_error("cannot open " + args.text_file);
    std::string line;
    while (std::getline(in, line)) {
      line = fireredpunc::Trim(line);
      if (!line.empty()) args.texts.push_back(line);
    }
  }
  if (args.texts.empty()) {
    args.texts = {"今天天气很好我们去公园散步吧",
                  "会议将在下午三点开始请大家准时参加",
                  "hello how are you i am fine thanks"};
  }
  return args;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const Args args = ParseArgs(argc, argv);
    AxRuntime runtime;
    PuncModel model(args.model_dir);
    for (const std::string& text : args.texts) {
      const std::string out = model.Punctuate(text, args.force);
      std::printf("IN : %s\n", text.c_str());
      std::printf("OUT: %s\n", out.c_str());
      if (args.stats) {
        const auto lat = model.Measure(text);
        std::printf("     median %.2f ms  p95 %.2f ms  min %.2f ms  max %.2f ms\n",
                    lat.median_ms, lat.p95_ms, lat.min_ms, lat.max_ms);
      }
    }
    return 0;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "error: %s\n", e.what());
    return 1;
  }
}
