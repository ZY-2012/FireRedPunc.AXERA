#pragma once

#include <regex>
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

inline std::string AddPunc(const std::vector<std::string>& tokens,
                           const std::vector<int>& preds,
                           const TokenDict& out_dict) {
  std::string txt;
  for (size_t i = 0; i < tokens.size(); ++i) {
    std::string token = tokens[i];
    const std::string tag = out_dict.Word(preds[i]);
    if (token.rfind("##", 0) == 0) {
      token = ReplaceAll(token, "##", "");
    } else if (std::regex_search(token, std::regex("[a-zA-Z0-9#]+")) &&
               i > 0 &&
               std::regex_search(tokens[i - 1], std::regex("[a-zA-Z0-9#]+"))) {
      if (out_dict.Word(preds[i - 1]) == " ") token = " " + token;
    }
    txt += token;
    if (tag != " ") txt += tag;
  }
  return ReplaceAll(txt, "  ", " ");
}

inline std::string RegexReplace(const std::string& text, const std::string& pattern,
                                const std::string& replacement) {
  return std::regex_replace(text, std::regex(pattern), replacement);
}

inline std::string RuleFix(const std::string& text_ori,
                           bool capitalize_first = true) {
  std::string txt = AsciiLowerCopy(text_ori);
  txt = RegexReplace(txt, "([a-z])，([a-z])", "$1, $2");
  txt = RegexReplace(txt, "([a-z])。([a-z])", "$1. $2");
  txt = RegexReplace(txt, "([a-z])？([a-z])", "$1? $2");
  txt = RegexReplace(txt, "([a-z])！([a-z])", "$1! $2");
  txt = RegexReplace(txt, "^([a-z]+)，", "$1,");
  txt = RegexReplace(txt, "^([a-z]+)。", "$1.");
  txt = RegexReplace(txt, "^([a-z]+)？", "$1?");
  txt = RegexReplace(txt, "^([a-z]+)！", "$1!");
  txt = RegexReplace(txt, "( [a-zA-Z']+)，$", "$1,");
  txt = RegexReplace(txt, "( [a-zA-Z']+)。$", "$1.");
  txt = RegexReplace(txt, "( [a-zA-Z']+)？$", "$1?");
  txt = RegexReplace(txt, "( [a-zA-Z']+)！$", "$1!");
  txt = RegexReplace(txt, "^i ", "I ");
  txt = RegexReplace(txt, "^i'm ", "I'm ");
  txt = RegexReplace(txt, "^i'd ", "I'd ");
  txt = RegexReplace(txt, "^i've ", "I've ");
  txt = RegexReplace(txt, "^i'll ", "I'll ");
  txt = RegexReplace(txt, " i ", " I ");
  txt = RegexReplace(txt, " i'm ", " I'm ");
  txt = RegexReplace(txt, " i'd ", " I'd ");
  txt = RegexReplace(txt, " i've ", " I've ");
  txt = RegexReplace(txt, " i'll ", " I'll ");

  auto cps = Utf8Decode(txt);
  if (capitalize_first && !cps.empty() && cps[0] >= 'a' && cps[0] <= 'z') {
    cps[0] = static_cast<uint32_t>(AsciiUpper(static_cast<char>(cps[0])));
  }

  std::string out;
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
