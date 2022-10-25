//
// Created by wanpeng.xie on 2022/5/6.
//

#ifndef SPARSE_DIST_MODEL_CONFIG_H
#define SPARSE_DIST_MODEL_CONFIG_H
#include <string>
#include <vector>
#include <memory>
#include <fstream>
#include <unordered_map>
#include <google/protobuf/text_format.h>
#include "config.pb.h"

namespace dist_linear_model {
inline std::string LoadFileToString(const std::string& path) {
  std::ifstream t(path);
  std::stringstream buffer;
  buffer << t.rdbuf();
  return buffer.str();
}

struct TrainConfig {
  float learning_rate_;
  float alpha_;
  float beta_;
  float l1_;
  float l2_;
  float fm_alpha_;
  float fm_beta_;
  float fm_l1_;
  float fm_l2_;
  explicit TrainConfig(const protos::FtrlConfig& ftrl_config);
};

struct FieldConfig {
  int32_t slot_id_;
  std::string name_;
  std::shared_ptr<TrainConfig> slot_config_;
  protos::VectorType vec_type_;
  int cross_field_;
  explicit FieldConfig(const protos::SlotConfig&);
};

struct ModelConfig {
  size_t dim_;
  size_t batch_size_;
  size_t async_step_;
  std::string load_model_path_;
  std::string load_inc_model_path_;
  std::string save_model_path_;
  std::string save_inc_model_path_;
  std::vector<std::string> train_path_list_;
  std::vector<std::string> test_path_list_;
  std::shared_ptr<TrainConfig> train_config_;
  std::vector<protos::SlotConfig> slot_lists_;
  std::unordered_map<int32_t, std::shared_ptr<protos::SlotConfig>> slot_maps_;
  //	model_config(std::string conf_path);
};

//bool NewModelConf(std::string path, std::shared_ptr<ModelConfig> model_config);

FieldConfig::FieldConfig(const protos::SlotConfig& slot_config) {
  slot_config_ = std::make_shared<TrainConfig>(slot_config.slot_config());
  slot_id_ = slot_config.slot_id();
  name_ = slot_config.name();
  vec_type_ = slot_config.vec_type();
  cross_field_ = slot_config.cross();
}

TrainConfig::TrainConfig(const protos::FtrlConfig& ftrl_config) {
  alpha_ = ftrl_config.alpha();
  beta_ = ftrl_config.beta();
  l1_ = ftrl_config.l1();
  l2_ = ftrl_config.l2();
  fm_alpha_ = ftrl_config.fm_alpha();
  fm_beta_ = ftrl_config.fm_beta();
  fm_l1_ = ftrl_config.fm_l1();
  fm_l2_ = ftrl_config.fm_l2();
}

bool NewModelConf(std::string path, std::shared_ptr<ModelConfig> model_config) {
  model_config->batch_size_ = ps::GetEnv("BATCH_SIZE", 16);

  protos::AllConfig config = protos::AllConfig();
  auto stat = google::protobuf::TextFormat::ParseFromString(
      LoadFileToString(path), &config);
  if (!stat) {
    return false;
  }
  model_config->dim_ = config.embedding_dim();

  model_config->train_config_ =
      std::make_shared<TrainConfig>(config.ftrl_config());

  model_config->save_model_path_ = config.save_model_path();
  model_config->save_inc_model_path_ = config.save_inc_model_path();
  model_config->load_model_path_ = config.load_model_path();
  model_config->load_inc_model_path_ = config.load_inc_model_path();
  model_config->train_path_list_ = std::vector<std::string>(
      config.train_path_list().begin(), config.train_path_list().end());
  model_config->test_path_list_ = std::vector<std::string>(
      config.test_path_list().begin(), config.test_path_list().end());

  model_config->slot_lists_ = std::vector<protos::SlotConfig>(
      config.slot_whitelist().begin(), config.slot_whitelist().end());
  for (auto slot : model_config->slot_lists_) {
    model_config->slot_maps_[slot.slot_id()] =
        std::make_shared<protos::SlotConfig>(slot);
  }

  // check config valid
  //		CHECK(model_config->load_model_path_.empty() || PathExist(model_config->load_model_path_)) << 		"load model path: " << model_config->load_model_path_ << " not exist";
  //		CHECK(model_config->load_inc_model_path_.empty() || PathExist(model_config->load_inc_model_path_)) <<
  //     "load inc model path: " << model_config->load_inc_model_path_ << "not exist";
  //
  //		CHECK(model_config->save_model_path_.empty() || !PathExist(model_config->save_model_path_)) << 		"save model path exist: " << model_config->save_model_path_; 		if (!model_config->save_model_path_.empty()) { 			CHECK(Mkdir(model_config->save_model_path_));
  //		}
  //
  //		CHECK(model_config->save_inc_model_path_.empty() || !PathExist(model_config->save_inc_model_path_)) << 		"save model path exist: " << model_config->save_inc_model_path_; 		if (!model_config->save_inc_model_path_.empty()) {
  //			CHECK(Mkdir(model_config->save_inc_model_path_));
  //		}
  //
  //		for (auto f : model_config->train_path_list_) {
  //			CHECK(PathExistWild(f)) << "training path: " << f << " not exist";
  //		}
  //		for (auto f:model_config->test_path_list_) {
  //			CHECK(PathExistWild(f)) << "test path: " << f << " not exist";
  //		}
  return true;
}
} // namespace dist_linear_model
#endif //SPARSE_DIST_MODEL_CONFIG_H
