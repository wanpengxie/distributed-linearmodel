//
// Created by xiewanpeng on 2022/10/25.
//

#ifndef DISTLM_INCLUDE_IO_FILES_H_
#define DISTLM_INCLUDE_IO_FILES_H_

#include <glob.h>
#include <sys/stat.h>

#include <fstream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "base/string_algo.h"
#include "shell.h"

namespace dist_linear_model {
static const uint64_t BUFFERSIZE = 100000;

bool is_hdfs(std::string path) {
  if (startsWith(path, "hdfs:")) {
    return true;
  }
  return false;
}

bool is_dir(const std::string& path) {
  struct stat s;
  if (stat(path.c_str(), &s) == 0) {
    if (s.st_mode & S_IFDIR) {
      return true;
    } else if (s.st_mode & S_IFREG) {
      return false;
    } else {
      // somthing else?
    }
  }
}

std::shared_ptr<FILE> LocalOpenRead(std::string path) {
  return FsOpenInternal(path, false, "r", BUFFERSIZE);
}

std::shared_ptr<FILE> HdfsOpenRead(std::string path) {
  if (endsWith(path, ".gz") || endsWith(path, ".snappy")) {
    path = string_format("hadoop fs -text \"%s\"", path.c_str());
  } else {
    path = string_format("hadoop fs -cat \"%s\"", path.c_str());
  }
  bool is_pipe = true;
  return FsOpenInternal(path, is_pipe, "r", BUFFERSIZE);
}

std::shared_ptr<FILE> OpenFile(std::string path) {
  if (is_hdfs(path)) {
    return HdfsOpenRead(path);
  } else {
    return LocalOpenRead(path);
  }
}

std::shared_ptr<FILE> HdfsOpenWrite(std::string path) {
  path = string_format("hadoop fs -put - \"%s\"", path.c_str());
  bool is_pipe = true;
  if (endsWith(path, ".gz\"")) {
    path = string_format("%s | %s", "gzip", path.c_str());
  }
  return FsOpenInternal(path, is_pipe, "w", BUFFERSIZE);
}

std::shared_ptr<FILE> LocalOpenWrite(std::string path) {
  return FsOpenInternal(path, false, "w", BUFFERSIZE);
}

std::shared_ptr<FILE> OpenWrite(std::string path) {
  if (is_hdfs(path)) {
    return HdfsOpenWrite(path);
  } else {
    return LocalOpenWrite(path);
  }
}

// string to uint64_t, split by any ^[0-9] char
void CustomAtoi(const char* str, uint64_t* val, int* index) {
  while (*str && *str >= '0' && *str <= '9') {
    *val = *val * 10 + (*str++ - '0');
    (*index)++;
  }
}

bool glob(const std::string& pattern, std::vector<std::string>& list) {
  using namespace std;
  // glob struct resides on the stack
  glob_t glob_result;
  memset(&glob_result, 0, sizeof(glob_result));

  int return_value = glob(pattern.c_str(), GLOB_TILDE, NULL, &glob_result);
  if (return_value != 0) {
    globfree(&glob_result);
    stringstream ss;
    ss << "glob() failed with return_value " << return_value << endl;
    return false;
  }

  // collect all the filenames into a std::list<std::string>
  for (size_t i = 0; i < glob_result.gl_pathc; ++i) {
    list.push_back(string(glob_result.gl_pathv[i]));
  }
  // cleanup
  globfree(&glob_result);
  // done
  return true;
}

std::vector<std::string> ListFile(const std::string& path) {
  if (path == "") {
    return {};
  }
  std::vector<std::string> list = {};
  if (is_hdfs(path)) {
    std::shared_ptr<FILE> pipe;
    pipe = ShellPopen("hadoop fs -ls " + path, "r");
    LineFileReader reader;
    const char delim = ' ';
    while (reader.GetLine(&*pipe)) {
      std::vector<std::string> line;
      splitString(std::string(reader.Get()), delim, line);
      if (line.size() >= 8) {
        list.push_back(line[line.size() - 1]);
      } else {
        // do nothing
      }
    }
  } else {
    if (is_dir(path)) {
      std::string npath(path);
      glob(rtrim(npath, "/") + "/*", list);
    } else {
      glob(path, list);
    }
  }
  return list;
}

// hdfs related
bool HdfsTouchz(const std::string& path) {
  std::string test = ShellGetCommandOutput(
      string_format("hadoop fs -touchz  %s ; echo $?", path.c_str()));
  if (string_trim(test) == "0") {
    return true;
  }
  return false;
}

bool HdfsExists(const std::string& path) {
  std::string test = ShellGetCommandOutput(
      string_format("hadoop fs -test -e %s ; echo $?", path.c_str()));
  if (string_trim(test) == "0") {
    return true;
  }
  return false;
}

bool LocalExist(const std::string& name) {
  std::ifstream f(name.c_str());
  return f.good();
}

bool PathExist(const std::string& path) {
  if (is_hdfs(path)) {
    return HdfsExists(path);
  } else {
    return LocalExist(path);
  }
}

bool PathExistWild(const std::string& path_pattern) {
  if (is_hdfs(path_pattern)) {
    return HdfsExists(path_pattern);
  } else {
    std::vector<std::string> lst;
    return glob(path_pattern, lst);
  }
}

bool Mkdir(const std::string& path) {
  if (PathExist(path)) {
    return true;
  }
  if (is_hdfs(path)) {
    std::string test = ShellGetCommandOutput(
        string_format("hadoop fs -mkdir -p %s ; echo $?", path.c_str()));
    if (string_trim(test) == "0") {
      return true;
    } else {
      return false;
    }
  } else {
    std::string test = ShellGetCommandOutput(
        string_format("mkdir -p %s ; echo $?", path.c_str()));
    if (string_trim(test) == "0") {
      return true;
    } else {
      return false;
    }
  }
}
}

#endif  // DISTLM_INCLUDE_IO_FILES_H_
