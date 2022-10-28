//
// Created by xiewanpeng on 2022/10/25.
//

#ifndef DISTLM_SRC_DIST_MODEL_H_
#define DISTLM_SRC_DIST_MODEL_H_
#include "ps/ps.h"
#include "base/utils.h"
#include "conf/model_config.h"
#include "data/dataloader.h"
#include "data/datawriter.h"


namespace dist_linear_model {
struct DistributedServer {
  DistributedServer(std::shared_ptr<ModelConfig> config = nullptr) {
    CHECK_NOTNULL(config);
    id_ = ps::MyRank();
    config_ = config;
    train_config_ = config_->train_config_;
    if (config_->model_name_ == "lr") {
      dim_ = 0;
    } else if (config_->model_name_ == "fm") {
      dim_ = config_ ->dim_;
    } else if (config_->model_name_ == "ffm"){
      int field = 0;
      for (auto x : config_->slot_lists_) {
        if (x.cross() > 0) {
          field = MAX(field, x.cross());
        }
      }
      dim_ = config_->dim_ * field;
    }
    alpha_ = train_config_->alpha_;
    beta_ = train_config_->beta_;
    l1_ = train_config_->l1_;
    l2_ = train_config_->l2_;
    LOG(INFO) << "server " << id_ << " start, dim=" << dim_ << std::endl;
  }

  void process(const ps::KVMeta &req_meta,
               const ps::KVPairs<float> &req_data,
               ps::KVServer<float> *server) {
    KVPairs<float> res;
    if (req_meta.cmd == TRAIN) {
      process_train_req(req_meta, req_data, res, false);
    } else if (req_meta.cmd == TEST) {
      process_train_req(req_meta, req_data, res, true);
    } else if (req_meta.cmd == LOAD) {
//      process_load_model(req_meta, req_data);
    } else {
      LOG(FATAL) << "wrong server cmd";
    }
    server->Response(req_meta, res);
  }

  void simple_process(const SimpleData& req, SimpleApp* app) {
    if (req.head == SAVE) {
      LOG(INFO) << "====start to save at server: " << id_;
//      process_save_model();
    }
    app->Response(req);
  }

  void process_train_req(const ps::KVMeta &req_meta,
                         const ps::KVPairs<float> &req_data,
                         KVPairs<float> &res,
                         bool eval = false) {
    size_t n = req_data.keys.size();
    size_t val_n = n * (dim_+1);

    if (!req_meta.pull) {
      CHECK_EQ(val_n, req_data.vals.size());
    } else {
      res.keys = req_data.keys;
      res.vals.resize(val_n);
    }

    for (size_t i = 0; i < n; ++i) {
      Key key = req_data.keys[i];

      std::shared_ptr<Paramter<float>> pm = get_parameter(key, eval);

      if (req_meta.pull) {
        for (size_t j = 0; j < dim_; j++) {
          res.vals[i*dim_+j] = pm->embedding_[j];
        }
      }

      if (req_meta.push) {
        int start_index = i * dim_ ;
        ftrl_update(pm, req_data.vals, start_index);
      }
    }
    return;
  }

  std::shared_ptr<Paramter<float>> get_parameter(uint64_t key, bool eval) {
    std::shared_ptr<Paramter<float>> pm;
    if (store_.find(key) == store_.end()) {
      pm = std::make_shared<Paramter<float>>();
      pm->embedding_.resize(dim_, 0);
      if (!eval) {
        pm->random_initial(dim_);
        store_[key] = pm;
      }
    } else {
      pm = store_[key];
    }
    return pm;
  }

  void ftrl_update(std::shared_ptr<Paramter<float>> pm, SArray<float> grads, int start_index) {
    for (size_t j = 0; j < dim_; j++) {
      float n = pm->embedding_n[j], z = pm->embedding_z[j], w = pm->embedding_[j];
      float g = grads[start_index + j];

      float sigma = (std::sqrt(n + g * g) - std::sqrt(n)) / alpha_;
      z += g - sigma * w;
      n += g * g;
      float sgn = 1.0;
      if (z < 0) {
        sgn = -1.0;
      }
      if (sgn * z < l1_) {
        w = 0;
      } else {
        w = -(z - sgn * l1_) /
            ((beta_ + std::sqrt(n)) / alpha_ + l2_);
      }
      pm->embedding_[j] = w;
      pm->embedding_n[j] = n;
      pm->embedding_z[j] = z;
    }
  }

  void sgd_update(std::shared_ptr<Paramter<float>> pm, SArray<float> grads, int start_index) {
    for (size_t j = 0; j < dim_; j++) {
      float g = grads[start_index + j];
      pm->embedding_[j] -= g * alpha_;
    }
  }

  void server_end() {
//    for (auto const &pair: store_) {
//      auto k = pair.first;
//      auto v = pair.second;
//    }
  }

//  void process_save_model() {
//    CHECK(!config_->save_model_path_.empty() || !config_->save_inc_model_path_.empty());
//    if (!config_->save_model_path_.empty()) {
//      std::string model_path = config_->save_model_path_ + "/" + "part-" + std::to_string(id_);
//      LOG(INFO) << "start saving model file: " << model_path;
//      save_model_file(model_path, false);
//      LOG(INFO) << "finish saving model file: " << model_path;
//    }
//    if (!config_->save_inc_model_path_.empty()) {
//      std::string model_path = config_->save_inc_model_path_ + "/" + "part-" + std::to_string(id_);
//      LOG(INFO) << "start saving inc model file: " << model_path;
//      save_model_file(model_path, true);
//      LOG(INFO) << "finish saving inc model file: " << model_path;
//    }
//  }

//  void save_model_file(const std::string &model_path, bool inc) {
//    std::shared_ptr<DataWriter> writer = std::make_shared<DataWriter>(model_path);
//    for (auto kv: store_) {
//      Key key = kv.first;
//      auto pm = kv.second;
//      std::string line = std::to_string(key) + '\t' + std::to_string(pm->slot_id) + '\t' +
//                         vector_to_line(pm->embedding_);
//      if (inc) {
//        line += '\t' + vector_to_line(pm->embedding_n);
//        line += '\t' + vector_to_line(pm->embedding_z);
//      }
//      writer->WriteLine(line, true);
//    }
//  }

//  void process_load_model(const ps::KVMeta &req_meta,
//                          const ps::KVPairs<float> &req_data) {
//    size_t n = req_data.keys.size();
//    size_t val_n = req_data.vals.size();
//    size_t dim_n = val_n / n;
//    CHECK(dim_n == dim_ || dim_n == dim_ * 3);
//    CHECK(req_data.keys.size() == req_data.lens.size());
//    for (size_t i = 0; i < n; ++i) {
//      Key key = req_data.keys[i];
//      int slot = req_data.lens[i];
//      size_t start_index = i * dim_n;
//      std::shared_ptr<Paramter<float>> pm = std::make_shared<Paramter<float>>();
//      pm->zero_initial(dim_);
//      // set value
//      pm->slot_id = slot;
//      for (size_t j = 0; j < dim_; ++j) {
//        pm->embedding_[j] = req_data.vals[start_index + j];
//      }
//      if (dim_n == dim_ * 3) {
//        for (size_t j = 0; j < dim_; ++j) {
//          pm->embedding_n[j] = req_data.vals[start_index + dim_ + j];
//        }
//        for (size_t j = 0; j < dim_; ++j) {
//          pm->embedding_z[j] = req_data.vals[start_index + 2 * dim_ + j];
//        }
//      }
//      store_[key] = pm;
//    }
//  }

 private:
  std::unordered_map<Key, std::shared_ptr<Paramter<float>>> store_;
  int id_;
  size_t dim_;
  std::shared_ptr<ModelConfig> config_;
  std::shared_ptr<TrainConfig> train_config_;
  float alpha_;
  float beta_;
  float l1_;
  float l2_;
};
}
#endif  // DISTLM_SRC_DIST_MODEL_H_
