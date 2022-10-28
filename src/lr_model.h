//
// Created by xiewanpeng on 2022/10/25.
//

#ifndef DISTLM_SRC_LR_MODEL_H_
#define DISTLM_SRC_LR_MODEL_H_
#include "ps/ps.h"
#include "model.h"

namespace dist_linear_model {
class LRModel : public Worker {
 public:
  LRModel(std::shared_ptr<ModelConfig> config, int app_id=0, int customer_id=0)
      : Worker(config, app_id, customer_id) {
            LOG(INFO) << "USING LR model" << std::endl;
            emb_dim_ = 0;
        };
  ~LRModel() {};
  void calc_score(std::vector<float>& scores, std::vector<std::shared_ptr<Sample>>& samples, WMap& weight_map) override;
  void calc_loss_and_gradient(std::vector<float>& gradient,
                              std::vector<std::shared_ptr<Sample>>& samples,
                              std::vector<Key>& keys, WMap& model) override;
};

void LRModel::calc_score(std::vector<float>& scores, std::vector<std::shared_ptr<Sample>>& samples, WMap& weight_map) {
  scores.resize(samples.size(), 0);
  float bias = weight_map[0]->embedding_[0]; // bias term
  for (size_t index=0; index<samples.size(); index++) {
    float s = bias;
    auto sample = samples[index];
    for (auto fid : sample->fea_ids_) {
      s += weight_map[fid]->embedding_[0];
    }
    scores[index] = sigmoid(s);
  }
}

void LRModel::calc_loss_and_gradient(std::vector<float>& gradient,
                                     std::vector<std::shared_ptr<Sample>>& samples,
                                     std::vector<Key>& keys, WMap& model) {
    std::vector<float> local_g(samples.size(), 0);
    auto dim = emb_dim_+1;
    float total_g = 0.0;
    std::vector<float> scores(samples.size(), 0);
    calc_score(scores, samples, model);
    for (size_t index=0; index<samples.size(); index++) {
      auto label = samples[index]->label_;
      auto score = scores[index];
      float g = 0.0;
      if (label == 0) {
        g = score;
      } else {
        g = score - 1.0;
      }
      total_g += g;
      for (auto k : samples[index]->fea_ids_) {
        CHECK_NE(model.find(k), model.end());
        auto gpm = model[k];
        gpm->grads_[0] += g; // first term as lr weight
      }
    }
    for (int i=0; i<keys.size(); i++) {
      auto k = keys[i];
      float grad = total_g;
      int batch = samples.size();
      if (k != BIAS) {
        grad = model[k]->grads_[0];
        batch = model[k]->show_;
      }
      for (int j=0; j<dim; j++) {
        gradient[i*dim + j] = grad / static_cast<float>(batch);
      }
    }
}
}
#endif  // DISTLM_SRC_LR_MODEL_H_
