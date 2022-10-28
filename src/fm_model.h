//
// Created by xiewanpeng on 2022/10/25.
//

#ifndef DISTLM_SRC_FM_MODEL_H_
#define DISTLM_SRC_FM_MODEL_H_
#include "base/utils.h"
#include "model.h"

namespace dist_linear_model {
struct FMModel : public Worker {
 public:
  FMModel(std::shared_ptr<ModelConfig> config, int app_id=0, int customer_id=0)
      : Worker(config, app_id, customer_id) {
        LOG(INFO) << "USING FM model" << std::endl;
      };
  ~FMModel() {};
  void calc_score(std::vector<float>& scores, std::vector<std::shared_ptr<Sample>>& samples, WMap& weight_map) override;
  void calc_score_fm(std::vector<float>& scores, std::vector<std::shared_ptr<Sample>>& samples, WMap& weight_map,
                     std::vector<std::vector<float>>& weight_sum);
  void calc_loss_and_gradient(std::vector<float>& gradient,
                              std::vector<std::shared_ptr<Sample>>& samples,
                              std::vector<Key>& keys, WMap& model) override;
};

void FMModel::calc_score(std::vector<float>& scores, std::vector<std::shared_ptr<Sample>>& samples, WMap& weight_map) {
  scores.resize(samples.size(), 0);
  std::vector<std::vector<float>> weight_sum;
  calc_score_fm(scores, samples, weight_map, weight_sum);
}

void FMModel::calc_score_fm(std::vector<float>& scores, std::vector<std::shared_ptr<Sample>>& samples, WMap& weight_map,
                   std::vector<std::vector<float>>& weight_sum) {
  float bias = weight_map[0]->embedding_[0];
  for (size_t index=0; index<samples.size(); index++) {
    float s = bias;
    std::vector<float> sample_weight_sum(emb_dim_, 0);
    float norm_sum = 0.0;
    float sum_norm = 0.0;

    auto sample = samples[index];

    for (auto fid : sample->fea_ids_) {
      auto weight_vec = weight_map[fid]->embedding_;
      s += weight_vec[0];
      for (size_t i=0; i<emb_dim_; i++) {
        auto w = weight_vec[i+1];
        sample_weight_sum[i] += w;
        norm_sum += w*w;
      }
    }
    for (size_t i=0; i<emb_dim_; i++) {
      sum_norm += sample_weight_sum[i] *  sample_weight_sum[i];
    }
    if (weight_sum.size() > 0) {
      weight_sum[index] = sample_weight_sum;
    }
    s += 0.5 * (sum_norm - norm_sum);
    scores[index] = sigmoid(s);
  }
  LOG(INFO) << "scores: " << to_line(scores);
}

void FMModel::calc_loss_and_gradient(std::vector<float>& gradient,
                                     std::vector<std::shared_ptr<Sample>>& samples,
                                     std::vector<Key>& keys, WMap& model) {
  std::vector<float> local_g(samples.size(), 0);
  auto dim = emb_dim_+1;
  float total_g = 0.0;

  std::vector<std::vector<float>> weight_sum;
  for (size_t i=0; i<samples.size(); i++) {
    std::vector<float> sample_weight_sum(emb_dim_, 0);
    weight_sum.push_back(sample_weight_sum);
  }
  std::vector<float> scores(samples.size(), 0);
  calc_score_fm(scores, samples, model, weight_sum);

  for (size_t index=0; index<samples.size(); index++) {
    auto label = samples[index]->label_;
    auto score = scores[index];
    auto sample_weight_sum = weight_sum[index];
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
      gpm->grads_[0] += g;
      for (size_t j=0; j<emb_dim_; j++) {
        gpm->grads_[j+1] += g * (sample_weight_sum[j] - gpm->embedding_[j+1]);
      }
    }
  }
  for (int i=0; i<keys.size(); i++) {
    auto k = keys[i];
    float grad = total_g;
    int batch = samples.size();
    auto gpm = model[k];
    if (k != BIAS) {
      float show = static_cast<float>(model[k]->show_);
      for (int j = 0; j < dim; j++) {
        gradient[i * dim + j] =  gpm->grads_[j] / show;
      }
    } else {
        // bias term gradient
        gradient[i*dim] = grad / static_cast<float>(batch);
    }
  }
  LOG(INFO) << "scores: " << to_line(scores);
  LOG(INFO) << "gradients: " << to_line(gradient);
}
}
#endif  // DISTLM_SRC_FM_MODEL_H_
