//
// Created by xiewanpeng on 2022/10/24.
//

#include <string>
#include "ps/ps.h"
#include "config.pb.h"
#include "model_config.h"


int main(int argc, char* argv[]) {
  std::string path = argv[1];
  auto conf = std::make_shared<dist_linear_model::ModelConfig>();
  dist_linear_model::NewModelConf(path, conf, false);
  LOG(INFO) << conf->batch_size_ << "|" << conf->train_config_->alpha_ << "|" << conf->train_path_list_[0] << "|"
            << conf->test_path_list_[0];

}