//
// Created by xiewanpeng on 2022/10/24.
//

#ifndef DISTLM_INCLUDE_IO_SHELL_H_
#define DISTLM_INCLUDE_IO_SHELL_H_

#include <fcntl.h>
#include <sys/syscall.h>
#include <sys/wait.h>
#include <sys/stat.h>
#include <unistd.h>
#include <stdio.h>
#include <memory>
#include <string>
#include <utility>
#include "ps/ps.h"
#include "line_file_reader.h"

namespace dist_linear_model {
inline bool& ShellVerboseInternal() {
  static bool x = false;
  return x;
}

bool ShellVerbose() { return ShellVerboseInternal(); }

void ShellSetVerbose(bool x) { ShellVerboseInternal() = x; }

// open file
std::shared_ptr<FILE> ShellFopen(const std::string& path,
                                 const std::string& mode) {
  FILE* fp;
  //  CHECK_EQ(fp, NULL);
  CHECK(fp = fopen(path.c_str(), mode.c_str()))
      << "path[" << path << "], mode[" << mode << "]";
  return {fp, [path](FILE* fp) { CHECK_EQ(0, fclose(fp)); }};
}

// Close all open file descriptors
void CloseOpenFdsInternal() {
  struct linux_dirent {
    long d_ino;
    off_t d_off;
    unsigned short d_reclen;
    char d_name[256];
  };

  int dir_fd;
  CHECK((dir_fd = open("/proc/self/fd", O_RDONLY)) >= 0);
  char buffer[sizeof(linux_dirent)];
  for (;;) {
    int bytes;
    //    CHECK((bytes = syscall(SYS_getdents, dir_fd, (linux_dirent*)buffer, sizeof(buffer))) >= 0);
    CHECK((bytes = syscall(SYS_getdents64, dir_fd, (linux_dirent*)buffer,
                           sizeof(buffer))) >= 0);
    if (bytes == 0) {
      break;
    }
    linux_dirent* entry;
    for (int offset = 0; offset < bytes; offset += entry->d_reclen) {
      entry = (linux_dirent*)(buffer + offset);
      int fd = 0;
      const char* s = entry->d_name;
      while (*s >= '0' && *s <= '9') {
        fd = fd * 10 + (*s - '0');
        s++;
      }
      if (s != entry->d_name && fd != dir_fd && fd >= 3) {
        close(fd);
      }
    }
  }
  close(dir_fd);
}

int ShellPopenForkInternal(const char* real_cmd, bool do_read, int parent_end,
                           int child_end) {
  int child_pid;
  // But vfork() is very dangerous. Be careful.
  CHECK((child_pid = vfork()) >= 0);
  // The following code is async signal safe
  // (No memory allocation, no access to global data, etc.)
  if (child_pid != 0) {
    return child_pid;
  }
  int child_std_end = do_read ? 1 : 0;
  close(parent_end);
  if (child_end != child_std_end) {
    CHECK(dup2(child_end, child_std_end) == child_std_end);
    close(child_end);
  }
  CloseOpenFdsInternal();
  CHECK(execl("/bin/bash", "bash", "-c", real_cmd, NULL) >= 0);
  exit(127);
}

// open pipe
std::shared_ptr<FILE> ShellPopen(const std::string& cmd,
                                 const std::string& mode) {
  bool do_read = mode == "r";
  bool do_write = mode == "w";
  CHECK(do_read || do_write);
  if (ShellVerbose()) {
    LOG(INFO) << "Opening pipe[" << cmd << "] with mode[" << mode << "]";
  }
  std::string real_cmd = "set -o pipefail; " + cmd;

  int pipe_fds[2];
  CHECK(pipe(pipe_fds) == 0);
  int parent_end = 0;
  int child_end = 0;
  if (do_read) {
    parent_end = pipe_fds[0];
    child_end = pipe_fds[1];
  } else if (do_write) {
    parent_end = pipe_fds[1];
    child_end = pipe_fds[0];
  }

  int child_pid =
      ShellPopenForkInternal(real_cmd.c_str(), do_read, parent_end, child_end);

  close(child_end);
  fcntl(parent_end, F_SETFD, FD_CLOEXEC);
  FILE* fp;
  CHECK((fp = fdopen(parent_end, mode.c_str())) != NULL);
  std::shared_ptr<FILE> ret_value(fp, [child_pid, cmd](FILE* fp) {
    if (ShellVerbose()) {
      LOG(INFO) << "Closing pipe[" << cmd << "]";
    }
    CHECK(fclose(fp) == 0);
    int wstatus, ret;
    do {
      CHECK((ret = waitpid(child_pid, &wstatus, 0)) >= 0 ||
            (ret == -1 && errno == EINTR));
    } while (ret == -1 && errno == EINTR);
    CHECK(wstatus == 0 || wstatus == (128 + SIGPIPE) * 256 ||
          (wstatus == -1 && errno == ECHILD));
    if (wstatus == -1 && errno == ECHILD) {
      LOG(WARNING) << "errno is ECHILD";
    }
  });
  return ret_value;
}

std::shared_ptr<FILE> FsOpenInternal(const std::string& path, bool is_pipe,
                                     const std::string& mode,
                                     size_t buffer_size) {
  std::shared_ptr<FILE> fp;
  if (!is_pipe) {
    fp = ShellFopen(path, mode);
  } else {
    fp = ShellPopen(path, mode);
  }
  if (buffer_size > 0) {
    char* buffer = new char[buffer_size];
    CHECK_EQ(0, setvbuf(&*fp, buffer, _IOFBF, buffer_size));
    fp = std::shared_ptr<FILE>(&*fp, [fp, buffer](FILE*) mutable {
      CHECK(fp.unique());
      fp = nullptr;
      delete[] buffer;
    });
  }
  return fp;
}

void ShellExecute(const std::string& cmd) { ShellPopen(cmd, "w"); }
//
std::string ShellGetCommandOutput(const std::string& cmd) {
  std::shared_ptr<FILE> pipe = ShellPopen(cmd, "r");
  LineFileReader reader;
  if (reader.GetDelim(&*pipe, 0)) {
    return reader.Get();
  }
  return "";
}
}

#endif  // DISTLM_INCLUDE_IO_SHELL_H_
