//
// Created by xiewanpeng on 2022/10/24.
//

#ifndef DISTLM_INCLUDE_IO_LINE_FILE_READER_H_
#define DISTLM_INCLUDE_IO_LINE_FILE_READER_H_

#define DECLARE_UNCOPYABLE(Class) \
private: \
    Class(const Class&); \
    Class& operator=(const Class&)

#include <cstdio>
#include "ps/base.h"

class LineFileReader  {
 public:
  LineFileReader();
  ~LineFileReader();
  char* GetLine(FILE* f);
  char* GetDelim(FILE* f, char delim);
  char* Get();
  size_t Length();

  DECLARE_UNCOPYABLE(LineFileReader);
 private:
  char* buffer_ = NULL;
  size_t buffer_size_ = 0;
  size_t length_ = 0;
};

LineFileReader::LineFileReader() {}
LineFileReader::~LineFileReader() {
  ::free(buffer_);
}
char* LineFileReader::GetLine(FILE* f) {
  return this->GetDelim(f, '\n');
}
char* LineFileReader::GetDelim(FILE* f, char delim) {
  ssize_t ret = ::getdelim(&buffer_, &buffer_size_, delim, f);
  if (ret >= 0) {
    if (ret >= 1 && buffer_[ret - 1] == delim) {
      buffer_[--ret] = 0;
    }
    length_ = (size_t)ret;
    return buffer_;
  } else {
    length_ = 0;
    CHECK(feof(f));
    return NULL;
  }
}
char* LineFileReader::Get() {
  return buffer_;
}
size_t LineFileReader::Length() {
  return length_;
}

#endif  // DISTLM_INCLUDE_IO_LINE_FILE_READER_H_
