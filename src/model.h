//
// Created by xiewanpeng on 2022/10/25.
//

#ifndef DISTLM_SRC_MODEL_H_
#define DISTLM_SRC_MODEL_H_
#include <unistd.h>

#include "ps/ps.h"
#include "data/dataloader.h"
#include "base/string_algo.h"
#include "base/utils.h"
#include "base/functions.h"
#include "conf/model_config.h"
#include "io/files.h"
#include "io/shell.h"
#include "metric/metric.h"

namespace dist_linear_model{
class Worker {
  public:
  Worker() {};
  Worker(std::shared_ptr<ModelConfig> config, int app_id=0, int customer_id=0) {
    kv_w_ = std::make_shared<ps::KVWorker<float>>(0, 0);

    using namespace std::placeholders;
    config_ = config;
    id_ = ps::MyRank();
    worker_numbers_ = ps::GetEnv("DMLC_NUM_WORKER", 1);
    emb_dim_ = config->dim_;
    async_step_ = config->async_step_;
    batch_size_ = config->batch_size_;
  }
  ~Worker() {}

  void Train();
  void Load();
  void Save();
  void Test();
  void parse_line(std::vector<Key>& keys, std::vector<float>& vals, std::vector<int>& slots,
                  const std::string& line, size_t dim);
  void test_file(std::string &test_path,
                 std::vector<int>& test_labels,
                 std::vector<float>& test_scores);
  void train_file(std::string& path);
  void load_file(std::string& path, bool inc);
  virtual void calc_score(std::vector<float>& scores,
                          std::vector<std::shared_ptr<Sample>>& samples,
                          WMap& weight_map) {};
  virtual void calc_loss_and_gradient(std::vector<float>& gradient,
                                      std::vector<std::shared_ptr<Sample>>& samples,
                                      std::vector<Key>& keys, WMap& model) {};

  int worker_numbers_ = 1;
  int id_;
  int core_num_;
  int batch_size_;
  int emb_dim_ = 1;
  int async_step_;
  std::string train_path_;
  std::shared_ptr<ps::KVWorker<float>> kv_w_;
  std::shared_ptr<ModelConfig> config_;
};

void Worker::Train()  {
  for (auto dir : config_->train_path_list_) {
    LOG(INFO) << "train files in: " << dir;
    auto file_list_all = ListFile(dir);
    std::vector<std::string> file_list;
    for (size_t i=0; i<file_list_all.size(); i++) {
      if (i%worker_numbers_ == id_) {
        file_list.push_back(file_list_all[i]);
      }
    }
    // train on single file
    for (auto p : file_list) {
      LOG(INFO) << "start to train: " << p << ", at worker: " << id_ << std::endl;
      sleep(1);
      train_file(p);
    }
  }
}

void Worker::Load() {
  //	auto recv_ids = kServerGroup;
  //	kv_w_->Wait(kv_w_->Request(sparse_dist::LOAD, "", recv_ids));
  bool inc=false;
  auto model_dir = config_->load_model_path_;
  CHECK(!model_dir.empty());
  auto model_files = ListFile(model_dir);
  for (int i=0; i<model_files.size(); ++i) {
    // split by node id
    if (i % NumWorkers() != id_) {
      continue;
    }
    auto path = model_files[i];
    LOG(INFO) << "worker=" << id_ << " start to load file: " << path;
    load_file(path, inc);
  }
}

void Worker::load_file(std::string& path, bool inc) {
  DataLoader reader = DataLoader(path, 1024);
  std::vector<std::string> lines;
  size_t dim = emb_dim_;
  if (inc) {
    dim = 3 * emb_dim_;
  }
  while (reader.GetLine(lines, 1024)) {
    std::vector<Key> keys;
    std::vector<int> slots;
    std::vector<float> vals;
    for (auto line : lines) {
      parse_line(keys, vals, slots, line, dim);
    }
    kv_w_->Wait(kv_w_->Push(keys, vals, slots, LOAD));
  }
}

void Worker::parse_line(std::vector<Key>& keys, std::vector<float>& vals, std::vector<int>& slots,
                          const std::string& line, size_t dim) {
  std::vector<std::string> parts;
  splitString(line, '\t', parts);
  CHECK(parts.size()  == dim + 2) << string_format("load file format: dim=%d, not fit current model: dim=%d",
                                                  parts.size() - 2, dim);
  uint64_t key=0;
  int slot=0;
  CHECK(stringToNumber(parts[0], key));
  CHECK(stringToNumber(parts[1], slot));
  keys.push_back(key);
  slots.push_back(slot);

  int index_shift = 2;
  if (parts.size() == emb_dim_ + index_shift) {
    for (int i=0; i<emb_dim_; ++i) {
      float w=0.0;
      CHECK(stringToNumber(parts[i+index_shift], w));
      vals.push_back(w);
    }
  } else if (parts.size() == 3 * emb_dim_ + index_shift) {
    for (int i=0; i<emb_dim_; ++i) {
      float w=0.0;
      CHECK(stringToNumber(parts[i+index_shift], w));
      vals.push_back(w);
    }
    index_shift += emb_dim_;
    for (int i=0; i<emb_dim_; ++i) {
      float w=0.0;
      CHECK(stringToNumber(parts[i+index_shift], w));
      vals.push_back(w);
    }
    index_shift += emb_dim_;
    for (int i=0; i<emb_dim_; ++i) {
      float w=0.0;
      CHECK(stringToNumber(parts[i+index_shift], w));
      vals.push_back(w);
    }
  }
}

void Worker::Save() {
  if (id_ == 0) {
    auto recv_ids = kServerGroup;
    auto ts = kv_w_->Request(SAVE, "", recv_ids);
    LOG(INFO) << "worker save: " << ts;
    kv_w_->Wait(ts);
  }
}

void Worker::Test() {
  if (MyRank() != 0 ) {
    return;
  }
  std::vector<int> labels;
  std::vector<float> scores;
  for (auto dir : config_->test_path_list_) {
    std::vector<std::string> files = ListFile(dir);
    for (auto f : files) {
      test_file(f, labels, scores);
    }
  }
  LOG(INFO) << "test sample size: " << labels.size();
  LOG(INFO) << "test auc: " << CalcAuc(labels, scores)
            << ", test loss: " << BinayLoss(labels, scores, "mean");
}

void Worker::train_file(std::string &path) {
  // todo: using file reader and multi thread
  auto reader = DataLoader(path, batch_size_);
  std::vector<std::shared_ptr<Sample>> samples;
  WMap local_model;
  int dimension =  emb_dim_ + 1;
  while (reader.GetSamples(samples)) {
    std::vector<Key> keys;
    local_model.clear();
    CollectKeys(samples, keys, local_model, dimension); // sorted and unique keys
    std::vector<float> w(keys.size() * (dimension));
    kv_w_->Wait(kv_w_->Pull(keys, &(w), nullptr, TRAIN));
    KVtoMap(keys, w, local_model);

    std::vector<float> gradients(w.size());
    calc_loss_and_gradient(gradients, samples, keys, local_model);
    kv_w_->Wait(kv_w_->Push(keys, gradients, {}, TRAIN));
  }
}

void Worker::test_file(std::string &test_path,
                         std::vector<int>& test_labels,
                         std::vector<float>& test_scores) {
  auto reader = DataLoader(test_path, 1024);
  std::vector<std::shared_ptr<Sample>> samples;
  WMap local_model;
  int dimension =  emb_dim_ + 1;

  while (reader.GetSamples(samples)) {
    std::vector<Key> keys;
    local_model.clear();
    CollectKeys(samples, keys, local_model, dimension);
    std::vector<float> w(keys.size() * dimension);
    kv_w_->Wait(kv_w_->Pull(keys, &(w), nullptr,TEST));
    KVtoMap(keys, w, local_model);

    // training samples with weight: return pctr and gradients
    std::vector<float> scores(samples.size());
    calc_score(scores, samples, local_model);

    for (int i=0; i<samples.size(); i++) {
      test_labels.push_back(samples[i]->label_);
      test_scores.push_back(scores[i]);
    }
  }
}
}
#endif  // DISTLM_SRC_MODEL_H_
