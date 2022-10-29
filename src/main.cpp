//
// Created by xiewanpeng on 2022/10/24.
//

#include <functional>
#include <random>

#include "io/files.h"
#include "conf/model_config.h"
#include "ps/base.h"
#include "ps/ps.h"
#include "server.h"
#include "model.h"
#include "lr_model.h"
#include "fm_model.h"
#include "ffm_model.h"
#include "scheduler.h"


using namespace ps;
using namespace dist_linear_model;

std::shared_ptr<Worker> create_model(std::shared_ptr<ModelConfig> config) {
  if (config->model_name_ == "lr") {
    return std::make_shared<LRModel>(config);
  } else if (config->model_name_ == "fm") {
    return std::make_shared<FMModel>(config);
  } else if (config->model_name_ == "ffm") {
    return std::make_shared<FFMModel>(config);
  }
}

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
  auto worker = create_model(config);

  using namespace std::placeholders;
  worker->kv_w_->SimpleApp::set_response_handle(std::bind(&Worker::simple_response_process, worker, _1, _2));

  if (!config->load_model_path_.empty()) {
    LOG(INFO) << "load model " << config->load_model_path_;
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
  if (!config->save_model_path_.empty()) {
    LOG(INFO) << "======start to save=====";
    worker->Save();
    ps::Postoffice::Get()->Barrier(0, kWorkerGroup);
    LOG(INFO) << "======end save=====";
  }
}


void start_scheduler(std::shared_ptr<ModelConfig> config) {
//  auto scheduler = new KVServer<float>(0);
//  scheduler->SimpleApp::set_request_handle(ReqHandle);

  auto scheduler = std::make_shared<Scheduler>(config);
  using namespace std::placeholders;
  scheduler->scheduler_->SimpleApp::set_request_handle(std::bind(&Scheduler::simple_req_handler, scheduler, _1, _2));
}

int main(int argc, char *argv[]) {
  ps::Start(0);
  std::string config_file = GetEnv("CONF", "");
  auto config = std::make_shared<ModelConfig>();;
  CHECK(LocalExist(config_file)) << "config file not exist, path=" << config_file;
  CHECK(NewModelConf(config_file, config, false));

  if (ps::IsScheduler()) {
    std::cout << "start schedule" << std::endl;
    start_scheduler(config);
  }

  if (ps::IsServer()) {
    std::cout << "start server: " << ps::MyRank() << std::endl;
    start_server(config);
  }

  if (ps::IsWorker()) {
    std::cout << "start worker: " << ps::MyRank() << std::endl;
    start_worker(config);
    std::cout << "start worker: " << ps::MyRank() << std::endl;
  }

  ps::Finalize(0, true);
  return 0;
}