//
// Created by xiewanpeng on 2022/10/24.
//

#ifndef DISTRIBUTED_LINEARMODEL_INCLUDE_METRIC_METRIC_H_
#define DISTRIBUTED_LINEARMODEL_INCLUDE_METRIC_METRIC_H_

namespace dist_linear_model {

struct LabelScore {
  int label_;
  float score_;
  LabelScore(){};
  LabelScore(int label, float score) {
    label_ = label;
    score_ = score;
  };

  bool operator<(const LabelScore& obj) const { return score_ < obj.score_; }
  bool operator>(const LabelScore& obj) const { return score_ > obj.score_; }

  bool operator<(const LabelScore* obj) const { return score_ < obj->score_; }
  bool operator>(const LabelScore* obj) const { return score_ > obj->score_; }
};

float CalcAuc(std::vector<int>& labels, std::vector<float>& scores) {
  std::vector<LabelScore> label_scores;
  for (int i = 0; i < labels.size(); i++) {
    label_scores.push_back(LabelScore(labels[i], scores[i]));
  }
  std::sort(label_scores.begin(), label_scores.end(), std::less<LabelScore>());

  int start = 0, n = labels.size();
  int pcount = 0, ncount = 0, tmpcount = 0, rank = 0;
  float rank_sum = 0.0;
  int i = 0;
  while (true) {
    if (start > n - 1) {
      break;
    }
    if (i == n ||
        (i > start && label_scores[i].score_ != label_scores[start].score_)) {
      float x = float(tmpcount) * (float(rank) + 0.5 * float(i - start + 1));
      rank_sum += x;
      rank += (i - start);
      start = i;
      tmpcount = 0;
    }
    if (i != n) {
      if (label_scores[i].label_ > 0) {
        pcount += 1;
        tmpcount += 1;
      } else {
        ncount += 1;
      }
    }
    i++;
  }
  float auc = 0.0;
  if (pcount * ncount == 0) {
    LOG(ERROR) << "all pos || all neg";
  }
  auc = (rank_sum - 0.5 * float(pcount) * float(pcount + 1)) / float(pcount) /
        float(ncount);
  return auc;
}

float BinayLoss(std::vector<int>& labels, std::vector<float>& scores,
                std::string summation = "sum") {
  float loss = 0.0;
  for (int i = 0; i < labels.size(); i++) {
    if (labels[i] == 0) {
      loss -= std::log(1.0 - scores[i]);
    } else {
      loss -= std::log(scores[i]);
    }
  }
  if (summation == "sum") {
    return loss;
  } else if (summation == "mean") {
    return loss / float(labels.size());
  } else {
    LOG(ERROR) << "unknow summation: " << summation;
    return 0.0;
  }
}

}

#endif  // DISTRIBUTED_LINEARMODEL_INCLUDE_METRIC_METRIC_H_
