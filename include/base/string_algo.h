//
// Created by xiewanpeng on 2022/10/25.
//

#ifndef DISTLM_INCLUDE_BASE_STRING_ALGO_H_
#define DISTLM_INCLUDE_BASE_STRING_ALGO_H_
#include <string>
#include <vector>
#include <sstream>
#include <stdexcept>
#include <memory>

template<typename ... Args>
std::string string_format( const std::string& format, Args ... args )
{
  int size_s = std::snprintf( nullptr, 0, format.c_str(), args ... ) + 1;
  if( size_s <= 0 ){ throw std::runtime_error( "Error during formatting." ); }
  auto size = static_cast<size_t>( size_s );
  std::unique_ptr<char[]> buf( new char[ size ] );
  std::snprintf( buf.get(), size, format.c_str(), args ... );
  return std::string( buf.get(), buf.get() + size - 1 );
}

// trim from end of string (right)
inline std::string& rtrim(std::string& s, const char* t = " \t\n\r\f\v")
{
  s.erase(s.find_last_not_of(t) + 1);
  return s;
}

// trim from beginning of string (left)
inline std::string& ltrim(std::string& s, const char* t = " \t\n\r\f\v")
{
  s.erase(0, s.find_first_not_of(t));
  return s;
}

// trim from both ends of string (right then left)
inline std::string& string_trim(std::string& s, const char* t = " \t\n\r\f\v")
{
  return ltrim(rtrim(s, t), t);
}
void splitString(const std::string &s, const char delim, std::vector<std::string> &elems) {
  std::stringstream ss(s);
  std::string item;
  while(std::getline(ss, item, delim)) {
    if (item.size() > 0) {
      elems.push_back(item);
    }
  }
}
bool stringToNumber(const std::string& line, uint64_t& number) {
  int index = 0;
  while (line[index]) {
    if (line[index] > '9' || line[index] < '0') {
      return false;
    }
    number = number * 10 + (line[index] - '0');
    index ++;
  }
  return true;
}

bool stringToNumber(const std::string& line, int& number) {
  int index = 0;
  while (line[index]) {
    if (line[index] > '9' || line[index] < '0') {
      return false;
    }
    number = number * 10 + (line[index] - '0');
    index ++;
  }
  return true;
}

bool stringToNumber(const std::string& line, float& number) {
  number = std::stof(line);
  return true;
}

bool endsWith(const std::string& str, const std::string& suffix)
{
  return str.size() >= suffix.size() && 0 == str.compare(str.size()-suffix.size(), suffix.size(), suffix);
}

bool startsWith(const std::string& str, const std::string& prefix)
{
  return str.size() >= prefix.size() && 0 == str.compare(0, prefix.size(), prefix);
}

#endif  // DISTLM_INCLUDE_BASE_STRING_ALGO_H_
