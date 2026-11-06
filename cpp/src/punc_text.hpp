#pragma once

#include <string>
#include <unordered_set>
#include <vector>

#include "bert_tokenizer.hpp"

namespace fireredpunc {

inline const std::unordered_set<uint32_t>& PuncAll() {
  static const std::unordered_set<uint32_t> kSet = {
      ',', '.', '?', '!', 0x3001, 0x3002, 0xFF1F, 0xFF01,
      0xFF0C, 0xFF0E, 0xFF61, 0xFF64};
  return kSet;
}

inline bool IsSentPunct(uint32_t cp) {
  return cp == '.' || cp == '?' || cp == '!' || cp == 0x3002 ||
         cp == 0xFF1F || cp == 0xFF01;
}

inline bool IsSkipPunct(uint32_t cp) {
  return cp == '?' || cp == '!' || cp == 0x3002 ||
         cp == 0xFF1F || cp == 0xFF01;
}

inline std::string StripMarks(const std::string& text) {
  const auto cps = Utf8Decode(text);
  std::string out;
  for (size_t i = 0; i < cps.size(); ++i) {
    const uint32_t cp = cps[i];
    bool keep = false;
    if (PuncAll().count(cp)) {
      const bool join = (cp == ',' || cp == '.' || cp == 0xFF0E) &&
                        i > 0 && i + 1 < cps.size() &&
                        cps[i - 1] >= '0' && cps[i - 1] <= '9' &&
                        cps[i + 1] >= '0' && cps[i + 1] <= '9';
      keep = join;
    }
    if (PuncAll().count(cp) && !keep) {
      out.push_back(' ');
    } else {
      out += Utf8Encode(cp);
    }
  }
  return out;
}

inline bool HasSentPunct(const std::string& text) {
  for (uint32_t cp : Utf8Decode(text)) {
    if (IsSkipPunct(cp)) return true;
  }
  return false;
}

inline bool IsAsciiLowerChar(char c) {
  return c >= 'a' && c <= 'z';
}

inline bool IsAsciiAlphaChar(char c) {
  return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z');
}

inline bool HasAlnumHash(const std::string& s) {
  for (unsigned char c : s) {
    if ((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') ||
        (c >= '0' && c <= '9') || c == '#') {
      return true;
    }
  }
  return false;
}

inline bool MatchCnPunct(const std::string& s, size_t pos, char* ascii) {
  if (pos + 2 >= s.size()) return false;
  const unsigned char a = static_cast<unsigned char>(s[pos]);
  const unsigned char b = static_cast<unsigned char>(s[pos + 1]);
  const unsigned char c = static_cast<unsigned char>(s[pos + 2]);
  if (a == 0xEF && b == 0xBC && c == 0x8C) { *ascii = ','; return true; }
  if (a == 0xE3 && b == 0x80 && c == 0x82) { *ascii = '.'; return true; }
  if (a == 0xEF && b == 0xBC && c == 0x9F) { *ascii = '?'; return true; }
  if (a == 0xEF && b == 0xBC && c == 0x81) { *ascii = '!'; return true; }
  return false;
}

inline std::string FixCnBetweenLetters(const std::string& s) {
  std::string out;
  out.reserve(s.size() + 8);
  for (size_t i = 0; i < s.size();) {
    char ascii = 0;
    if (i + 4 < s.size() && IsAsciiLowerChar(s[i]) &&
        MatchCnPunct(s, i + 1, &ascii) && IsAsciiLowerChar(s[i + 4])) {
      out.push_back(s[i]);
      out.push_back(ascii);
      out.push_back(' ');
      out.push_back(s[i + 4]);
      i += 5;
    } else {
      out.push_back(s[i++]);
    }
  }
  return out;
}

inline std::string FixCnAtStart(const std::string& s) {
  size_t j = 0;
  while (j < s.size() && IsAsciiLowerChar(s[j])) ++j;
  char ascii = 0;
  if (j > 0 && MatchCnPunct(s, j, &ascii)) {
    return s.substr(0, j) + ascii + s.substr(j + 3);
  }
  return s;
}

inline std::string FixCnAtEnd(const std::string& s) {
  if (s.size() < 3) return s;
  const size_t pos = s.size() - 3;
  char ascii = 0;
  if (!MatchCnPunct(s, pos, &ascii)) return s;
  size_t k = pos;
  while (k > 0 && (IsAsciiAlphaChar(s[k - 1]) || s[k - 1] == '\'')) --k;
  if (k < pos && k > 0 && s[k - 1] == ' ') {
    return s.substr(0, pos) + ascii;
  }
  return s;
}

inline std::string ReplacePrefix(std::string s, const std::string& from,
                                 const std::string& to) {
  if (s.rfind(from, 0) == 0) s.replace(0, from.size(), to);
  return s;
}

inline std::string AddPunc(const std::vector<std::string>& tokens,
                           const std::vector<int>& preds,
                           const TokenDict& out_dict) {
  std::string txt;
  for (size_t i = 0; i < tokens.size(); ++i) {
    std::string token = tokens[i];
    const std::string tag = out_dict.Word(preds[i]);
    if (token.rfind("##", 0) == 0) {
      token = ReplaceAll(token, "##", "");
    } else if (HasAlnumHash(token) && i > 0 &&
               HasAlnumHash(tokens[i - 1])) {
      if (out_dict.Word(preds[i - 1]) == " ") token = " " + token;
    }
    txt += token;
    if (tag != " ") txt += tag;
  }
  return ReplaceAll(txt, "  ", " ");
}

inline std::string RuleFix(const std::string& text_ori,
                           bool capitalize_first = true) {
  std::string txt = FixCnBetweenLetters(AsciiLowerCopy(text_ori));
  txt = FixCnAtStart(txt);
  txt = FixCnAtEnd(txt);

  txt = ReplacePrefix(txt, "i ", "I ");
  txt = ReplacePrefix(txt, "i'm ", "I'm ");
  txt = ReplacePrefix(txt, "i'd ", "I'd ");
  txt = ReplacePrefix(txt, "i've ", "I've ");
  txt = ReplacePrefix(txt, "i'll ", "I'll ");
  txt = ReplaceAll(txt, " i ", " I ");
  txt = ReplaceAll(txt, " i'm ", " I'm ");
  txt = ReplaceAll(txt, " i'd ", " I'd ");
  txt = ReplaceAll(txt, " i've ", " I've ");
  txt = ReplaceAll(txt, " i'll ", " I'll ");

  if (capitalize_first && !txt.empty() && IsAsciiLowerChar(txt[0])) {
    txt[0] = AsciiUpper(txt[0]);
  }

  const auto cps = Utf8Decode(txt);
  std::string out;
  out.reserve(txt.size() + 4);
  for (size_t i = 0; i < cps.size(); ++i) {
    out += Utf8Encode(cps[i]);
    if (!IsSentPunct(cps[i])) continue;
    size_t j = i + 1;
    while (j < cps.size() && IsWhitespace(cps[j])) ++j;
    if (j > i + 1 && j < cps.size() && cps[j] >= 'a' && cps[j] <= 'z') {
      out.push_back(' ');
      out += Utf8Encode(static_cast<uint32_t>(
          AsciiUpper(static_cast<char>(cps[j]))));
      i = j;
    }
  }
  return out;
}

}  // namespace fireredpunc
