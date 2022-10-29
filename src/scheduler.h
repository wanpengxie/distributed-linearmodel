//
// Created by xiewanpeng on 2022/10/29.
//

#ifndef DISTLM_SRC_SCHEDULER_H_
#define DISTLM_SRC_SCHEDULER_H_

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

namespace dist_linear_model {
class Scheduler {
 public:
  Scheduler(std::shared_ptr<ModelConfig> config){
      scheduler_ = std::make_shared<KVServer<float>>(0);
      not_initial_ready_ = false;
      mu_.lock();
      for (auto p : config->train_path_list_) {
        auto files = ListFile(p);
        for (auto f : files) {
          train_list_.push_back(f);
          LOG(INFO) << "add to joblist: " << f;
        }
      }
      not_initial_ready_ = true;
      mu_.unlock();
  };

  void simple_req_handler(const SimpleData& req, SimpleApp* app) {
    // LOG(INFO) << req.body;
    if (req.head == LOGGER) {
      LOG(INFO) << req.body;
      app->Response(req);
      return ;
    }
    if (req.head == JOB) {
      mu_.lock();
      auto new_req = SimpleData(req);
      std::string body;
      if (train_list_.empty()) {
        new_req.head = JOBEND;
        body = "NULL";
      } else {
        body = train_list_.back();
        train_list_.pop_back();
      }
      mu_.unlock();
      app->Response(new_req, body);
    }
  }

//  void read_lists(std::vector<std::string> dirs) {
//    mu_.lock();
//    for (auto p : dirs) {
//      auto files = ListFile(p);
//      for (auto f : files) {
//        train_list_.push_back(f);
//        LOG(INFO) << "add to joblist: " << f;
//      }
//    }
//    mu_.unlock();
//  }
  std::shared_ptr<KVServer<float>> scheduler_;
  std::vector<std::string> train_list_;
  std::vector<std::string> model_list_;
  bool not_initial_ready_;
  mutable std::mutex mu_;
};
}
#endif  // DISTLM_SRC_SCHEDULER_H_