//
// Created by xiewanpeng on 2022/10/24.
//
#include "base/utils.h"
#include "io/files.h"
#include "io/shell.h"
#include "conf/model_config.h"
#include "ps/ps.h"  // using dmlc log

using namespace dist_linear_model;
int main(int argc, char *argv[]) {
  // dmlc logger
  std::string path = argv[1];
  auto config = std::make_shared<ModelConfig>();
  dist_linear_model::NewModelConf(path, config, false);
  auto files = dist_linear_model::ListFile(config->train_path_list_[0]);
  LOG(INFO) << dist_linear_model::to_line(files);
}