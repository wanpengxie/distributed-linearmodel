//
// Created by xiewanpeng on 2022/10/25.
//
#ifndef DISTLM_INCLUDE_DATA_DATALOADER
#define DISTLM_INCLUDE_DATA_DATALOADER

#include <string>
#include <vector>

#include "base/utils.h"
#include "io/files.h"

namespace dist_linear_model {
struct DataLoader {
  DataLoader(const std::string& path, int batch_size) {
    batch_size_ = batch_size;
    path_ = path;
    fp_ = OpenFile(path);
    reader_ = std::make_shared<LineFileReader>();
  }
  bool GetSamples(std::vector<std::shared_ptr<Sample> >& samples);
  bool parseSample(std::string& line, std::shared_ptr<Sample> sample);
  bool GetLine(std::string& line);
  bool GetLine(std::vector<std::string>& lines, int batch);
  int batch_size_;
  std::string path_;
  std::shared_ptr<FILE> fp_;
  std::shared_ptr<LineFileReader> reader_;
};

bool DataLoader::GetLine(std::string& line) {
  if (reader_->GetLine(fp_.get()) != NULL) {
    line.assign(reader_->Get(), reader_->Length());
    return true;
  } else {
    return false;
  }
}

bool DataLoader::GetLine(std::vector<std::string>& lines, int batch) {
  lines.clear();
  while (reader_->GetLine(fp_.get()) != NULL) {
    std::string line;
    line.assign(reader_->Get(), reader_->Length());
    lines.push_back(line);
    if (lines.size() >= batch) {
      break;
    }
  }
  return (lines.size() > 0);
}

bool DataLoader::GetSamples(std::vector<std::shared_ptr<Sample> >& samples) {
  samples.clear();
  std::string line;
  while (reader_->GetLine(fp_.get()) != NULL) {
    line.assign(reader_->Get(), reader_->Length());
    std::shared_ptr<Sample> sample(new Sample());
    if (parseSample(line, sample)) {
      samples.push_back(sample);
    }
    if (samples.size() >= batch_size_) {
      break;
    }
  }
  return (samples.size() > 0);
}

bool DataLoader::parseSample(std::string& line,
                             std::shared_ptr<Sample> sample) {
  if (line.empty()) return false;
  int index = 0;
  int size = line.size();

  uint64_t label = 0;
  CustomAtoi(line.c_str() + index, &label, &index);
  index++;
  sample->label_ = label;

  while (index < size) {
    uint64_t slot = 0, fid = 0;
    CustomAtoi(line.c_str() + index, &slot, &index);
    ++index;
    CustomAtoi(line.c_str() + index, &fid, &index);
    ++index;
    sample->fea_ids_.push_back(fid);
    sample->slot_ids_.push_back(slot);
  }
  return true;
}
}
#endif  // DISTLM_INCLUDE_DATA_DATALOADER
