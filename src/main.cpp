//
// Created by xiewanpeng on 2022/10/24.
//

#include <functional>
#include <random>

#include "base/utils.h"
#include "io/files.h"
#include "model_config.h"
#include "ps/base.h"
#include "ps/ps.h"
#include "lr_model.h"
#include "server.h"


using namespace ps;
using namespace dist_linear_model;

// server is the same for all model
void start_server(std::shared_ptr<ModelConfig> config) {
  auto server = new KVServer<float>(0);
  auto dp = new DistributedServer(config);

  using namespace std::placeholders;

  server->set_request_handle(std::bind(&DistributedServer::process, dp, _1, _2, _3));
  server->SimpleApp::set_request_handle(std::bind(&DistributedServer::simple_process, dp, _1, _2));
  RegisterExitCallback([server, dp](){
    dp->server_end();
    delete server; });
}

// worker depends on different model
void start_worker(std::shared_ptr<ModelConfig> config) {
  auto epoch = GetEnv("EPOCH", 1);
  auto worker = new LRWorker(config);
  if (!config->load_inc_model_path_.empty() || !config->load_model_path_.empty()) {
    LOG(INFO) << "load model " << config->load_inc_model_path_;
    worker->Load();
    ps::Postoffice::Get()->Barrier(0, kWorkerGroup);
    LOG(INFO) << "======finish load, start next step=====" ;
  }

  if (!config->train_path_list_.empty()) {
    for (int i=0; i<epoch; i++) {
      LOG(INFO) << "======start train epoch: " << i << " ===========" ;
      worker->Train();
    }
    ps::Postoffice::Get()->Barrier(0, kWorkerGroup);
    LOG(INFO) << "======finish train, start to test=====" ;
  }
  if (!config->test_path_list_.empty()) {
    LOG(INFO) << "======start to test=====" ;
    worker->Test();
    ps::Postoffice::Get()->Barrier(0, kWorkerGroup);
    LOG(INFO) << "======end to test=====" ;
  }
  if (!config->save_inc_model_path_.empty() || !config->save_model_path_.empty()) {
    LOG(INFO) << "======start to save=====";
    worker->Save();
    ps::Postoffice::Get()->Barrier(0, kWorkerGroup);
    LOG(INFO) << "======end save=====";
  }
}

int main(int argc, char *argv[]) {
  ps::Start(0);
  std::string config_file = GetEnv("CONF", "");
  auto config = std::make_shared<ModelConfig>();;
  CHECK(LocalExist(config_file)) << "config file not exist, path=" << config_file;
  CHECK(NewModelConf(config_file, config));

  if (ps::IsScheduler()) {
    std::cout << "test schedule" << std::endl;
  }

  if (ps::IsServer()) {
    start_server(config);
  }

  if (ps::IsWorker()) {
    std::cout << "test worker: " << ps::MyRank() << std::endl;
    start_worker(config);
  }

  ps::Finalize(0, true);
  return 0;
}