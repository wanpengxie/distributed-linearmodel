//
// Created by xiewanpeng on 2022/10/24.
//

#ifndef DISTLM_INCLUDE_BASE_FUNCTIONS_H_
#define DISTLM_INCLUDE_BASE_FUNCTIONS_H_
#include <sys/stat.h>
#include <stdio.h>
#include <glob.h>
#include <string.h>
#include <stdexcept>
#include <sstream>
#include <vector>
#include <string>
#include <memory>
#include <fstream>

namespace dist_linear_model {
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
}
#endif  // DISTLM_INCLUDE_BASE_FUNCTIONS_H_
