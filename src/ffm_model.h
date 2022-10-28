//
// Created by xiewanpeng on 2022/10/25.
//

#ifndef DISTLM_SRC_FFM_MODEL_H_
#define DISTLM_SRC_FFM_MODEL_H_
#include "base/utils.h"
#include "model.h"

namespace dist_linear_model {
struct FFMModel : public Worker {
 public:
  FFMModel(std::shared_ptr<ModelConfig> config, int app_id=0, int customer_id=0)
      : Worker(config, app_id, customer_id) {
        LOG(INFO) << "USING FFM model" << std::endl;
        // initial ffm model
        single_field_dim_ = config->dim_;
        field_size_ = 0;
        for (auto feature : config->slot_lists_) {
          slot_to_field_[feature.slot_id()] = feature.cross();
          field_size_ = MAX(field_size_, feature.cross());
        }
        emb_dim_ = field_size_ * single_field_dim_;
      };
  ~FFMModel(){};
  float calc_ffm_inner_product(std::vector<float> ffm_vec, float norm_sum);
  void calc_score(std::vector<float>& scores, std::vector<std::shared_ptr<Sample>>& samples, WMap& weight_map) override;
  void calc_score_ffm(std::vector<float>& scores, std::vector<std::shared_ptr<Sample>>& samples, WMap& weight_map,
                     std::vector<std::vector<float>>& weight_sum);
  void calc_loss_and_gradient(std::vector<float>& gradient,
                              std::vector<std::shared_ptr<Sample>>& samples,
                              std::vector<Key>& keys, WMap& model) override;
  std::unordered_map<int, int> slot_to_field_;
  int single_field_dim_;
  int field_size_;
};

void FFMModel::calc_score(std::vector<float>& scores, std::vector<std::shared_ptr<Sample>>& samples, WMap& weight_map) {
  scores.resize(samples.size(), 0);
  std::vector<std::vector<float>> weight_sum;
  calc_score_ffm(scores, samples, weight_map, weight_sum);
}

void FFMModel::calc_score_ffm(std::vector<float>& scores, std::vector<std::shared_ptr<Sample>>& samples, WMap& weight_map,
                   std::vector<std::vector<float>>& weight_sum) {
  float bias = weight_map[BIAS]->embedding_[0];
  std::vector<float> ffm_grad_vec(emb_dim_ + 1, 0.0);

  for (size_t index=0; index<samples.size(); index++) {
    auto sample = samples[index];
    std::vector<float> ffm_weight_vec(emb_dim_ * field_size_, 0.0);

    float norm_sum = 0.0;
    float s = bias;

    for (size_t i=0; i<sample->fea_ids_.size(); i++) {
      uint64_t fid = sample->fea_ids_[i];
      uint64_t slot = sample->slot_ids_[i];
      int field = slot_to_field_[slot];

      auto weight_vec = weight_map[fid]->embedding_;
      CHECK_EQ(weight_vec.size(), emb_dim_ + 1);
      s += weight_vec[0];
      if (field == 0) {
        continue ;
      }
      int field_start_index = emb_dim_ * (field-1);
      for (size_t j=0; j<emb_dim_; j++) {
        float w = weight_vec[1+j];
        ffm_weight_vec[field_start_index+j] += w;
        if ((j/single_field_dim_) == (field-1)) {
          norm_sum += w*w;
        }
      }
    }
    s = s + calc_ffm_inner_product(ffm_weight_vec, norm_sum) - norm_sum * 0.5;
    scores[index] = sigmoid(s);
    if (weight_sum.size() > 0) {
      weight_sum[index] = ffm_weight_vec;
    }
  }
  LOG(INFO) << "scores: " << to_line(scores);
}

float FFMModel::calc_ffm_inner_product(std::vector<float> ffm_vec, float norm_sum) {
  CHECK_EQ(ffm_vec.size(), emb_dim_*field_size_);
  float s = 0.0;
  for (size_t field_i=0; field_i<field_size_; field_i++) {
    int field_i_vec_start_index = field_i * emb_dim_;
    for (size_t field_j=field_i; field_j<field_size_; field_j++) {
      int field_j_vec_start_index = field_j * emb_dim_;
      int field_i_start = field_i_vec_start_index + field_j * single_field_dim_;
      int field_j_start = field_j_vec_start_index + field_i * single_field_dim_;
      float val = 1.0;
      if (field_i == field_j) {
        val = 0.5;
      }
      for (size_t i=0; i<single_field_dim_; i++) {
        s += ffm_vec[field_i_start+i] * ffm_vec[field_j_start+i] * val;
      }
    }
  }
  return s;
}

void FFMModel::calc_loss_and_gradient(std::vector<float>& gradient,
                                     std::vector<std::shared_ptr<Sample>>& samples,
                                     std::vector<Key>& keys, WMap& model) {
  std::vector<float> local_g(samples.size(), 0);
  auto dim = emb_dim_+1;
  float total_g = 0.0;

  std::vector<std::vector<float>> weight_sum;
  for (size_t i=0; i<samples.size(); i++) {
    std::vector<float> sample_weight_sum(0, 0);
    weight_sum.push_back(sample_weight_sum);
  }
  std::vector<float> scores(samples.size(), 0);
  calc_score_ffm(scores, samples, model, weight_sum);

  for (size_t index=0; index<samples.size(); index++) {
    auto label = samples[index]->label_;
    auto score = scores[index];
    auto ffm_field_vec = weight_sum[index];
    float g = 0.0;
    if (label == 0) {
      g = score;
    } else {
      g = score - 1.0;
    }
    total_g += g;
    for (size_t fea_index=0; fea_index < samples[index]->fea_ids_.size(); fea_index++) {
      auto key = samples[index]->fea_ids_[fea_index];
      auto slot = samples[index]->slot_ids_[fea_index];
      auto field = slot_to_field_[slot];
      auto feature_field = field -1;
      CHECK_NE(model.find(key), model.end());
      auto gpm = model[key];
      gpm->grads_[0] += g;
      if (field == 0) {
        continue ;
      }

      for (size_t j=0; j<emb_dim_; j++) {
        int field_i = j / single_field_dim_;
        int index_j = j % single_field_dim_;
        int daul_ffm_start_index = field_i * emb_dim_ + feature_field * single_field_dim_;
        gpm->grads_[j+1] += g * ffm_field_vec[daul_ffm_start_index + index_j];

        // for fm part, remove self interaction
        if (field_i == feature_field) {
          gpm->grads_[j+1] -= g * gpm->embedding_[j+1];
        }
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
#endif  // DISTLM_SRC_FFM_MODEL_H_
