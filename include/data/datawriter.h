//
// Created by xiewanpeng on 2022/10/25.
//


#include <string>

#include "base/utils.h"
#include "io/files.h"

namespace dist_linear_model {
struct DataWriter {
  DataWriter(const std::string& path) {
    path_ = path;
    fp_ = OpenWrite(path);
    writer_ = std::make_shared<LineFileReader>();
  }
  void WriteLine(std::string& line, bool new_line=false);
  std::string path_;
  std::shared_ptr<FILE> fp_;
  std::shared_ptr<LineFileReader> writer_;
};

void DataWriter::WriteLine(std::string &line, bool new_line) {
  CHECK(std::fwrite(line.c_str(), line.length(), 1, fp_.get()) == 1)
      << "write line fail: " << line << " to file: " << path_;
  if (new_line) {
    CHECK(std::fwrite("\n", sizeof(char), 1, fp_.get()) == 1);
  }
}
}