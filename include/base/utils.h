//
// Created by xiewanpeng on 2022/10/24.
//

#ifndef DISTRIBUTED_LINEARMODEL_UTILS_H
#define DISTRIBUTED_LINEARMODEL_UTILS_H
#define MAX_EXP_NUM 50.
#define MIN_SIGMOID (10e-8)
#define MAX_SIGMOID (1. - 10e-8)

#include <stdlib.h>
#include <random>
//#include <unistd.h>
#include <time.h>
#include "ps/ps.h"

namespace dist_linear_model {

using namespace ps;

const static uint64_t BIAS = 0;
const static int BIASID = 1000;

enum MODELCOMMANDS {
  UNKNOWN = 1,
  TRAIN = 2,
  TEST = 3,
  LOAD = 4,
  LOADINC = 5,
  SAVE = 6,
  LOGGER = 7,
};

const std::string currentDateTime() {
  time_t now = time(0);
  struct tm tstruct;
  char buf[80];
  tstruct = *localtime(&now);
  strftime(buf, sizeof(buf), "%Y-%m-%d %X", &tstruct);

  return buf;
}

template <typename V>
struct Paramter {
  int show_ = 0;
  int slot_id = 0;
  std::vector<V> embedding_;
  std::vector<V> embedding_n;
  std::vector<V> embedding_z;

  void random_initial(int n) {
    embedding_n.resize(n, 0.0);
    embedding_z.resize(n, 0.0);
    embedding_.resize(n, 0.0);
    for (int i=1; i<n; i++) {
      float w = rand()/(RAND_MAX * 1.0);
      embedding_[i] = (w-0.5);
    }
  }

  void random_initial(int n, float norm) {
    embedding_n.resize(n, 0.0);
    embedding_z.resize(n, 0.0);
    embedding_.resize(n, 0.0);
    for (int i=1; i<n; i++) {
      float w = rand()/(RAND_MAX * 1.0);
      embedding_[i] = (w-0.5)/norm;
    }
  }

  void zero_initial(int n) {
    embedding_n.resize(n, 0.0);
    embedding_z.resize(n, 0.0);
    embedding_.resize(n, 0.0);
  }
};

template <typename V>
struct ParamterWeight {
  int show_ = 0;
  int slot_id_ = 0;
  std::vector<V> embedding_;
  std::vector<V> grads_;

  ParamterWeight(){};

  ParamterWeight(int n) {
    embedding_.resize(n, 0);
    grads_.resize(n, 0);
  };
};

typedef std::unordered_map<Key, std::shared_ptr<ParamterWeight<float>>> WMap;
typedef std::unordered_map<Key, std::shared_ptr<Paramter<float>>> MMap;

struct Sample {
  int label_;
  std::vector<int> labels_;
  std::vector<Key> fea_ids_;
  std::vector<int> slot_ids_;

  Sample() {
    label_ = 0;
    fea_ids_.reserve(50);
  }
};

void CollectKeys(std::vector<std::shared_ptr<Sample>> &samples,
                        std::vector<Key> &keys,
                        WMap &model, int emb) {
  model.clear();
  keys.push_back(BIAS);
  auto bias = std::make_shared<ParamterWeight<float>>(emb);
  bias->show_ = samples.size();
  model[BIAS] = bias;

  for (auto const &sample : samples) {
    for (size_t i(0); i < sample->fea_ids_.size(); ++i) {
      auto fid = sample->fea_ids_[i];
      if (model.find(fid) == model.end()) {
        auto pms = std::make_shared<ParamterWeight<float>>(emb);
        model[fid] = pms;
        keys.push_back(fid);
      }
      model[fid]->show_ += 1;
    }
  }
  std::sort(keys.begin(), keys.end());
}

template <typename V>
void KVtoMap(std::vector<Key> &keys, std::vector<V> &weights,
             std::unordered_map<Key, std::shared_ptr<ParamterWeight<V>>> model) {
  size_t dim = weights.size() / keys.size();
  for (size_t i = 0; i < keys.size(); i++) {
    size_t start_index = i * dim;
    Key key = keys[i];
    CHECK_NE(model.find(key), model.end());
    std::shared_ptr<ParamterWeight<V>> pms = model[key];
    CHECK_EQ(pms->embedding_.size(), dim);
    CHECK_EQ(pms->grads_.size(), dim);
    for (size_t j = 0; j < dim; j++) {
      pms->embedding_[j] = weights[start_index + j];
    }
  }
}

// math
inline float safe_exp(float x) {
  auto max_exp = static_cast<float>(MAX_EXP_NUM);
  return std::exp(std::max(std::min(x, max_exp), -max_exp));
}

inline float sigmoid(float x) {
  float one = 1.;
  float res = one / (one + safe_exp(-x));
  auto min_sigmoid = static_cast<float>(MIN_SIGMOID);
  auto max_sigmoid = static_cast<float>(MAX_SIGMOID);
  return std::max(std::min(res, max_sigmoid), min_sigmoid);
}

template <typename V>
std::string to_line(const V* input, int size,
                           const char delim = ' ') {
  if (size == 0) {
    return "";
  }
  std::stringstream ss;
  for (int i = 0; i < size-1; ++i) ss << input[i] << ",";
  ss << input[size-1];
  return ss.str();
}

template <typename V>
std::string to_line(std::vector<V> vec) {
  return to_line(vec.data(), vec.size(), ',');
}

template <typename V>
std::string print_vector(std::vector<V> const &input) {
  return to_line(input, input.size(), ',');
}
}
#endif //DISTRIBUTED_LINEARMODEL_UTILS_H
