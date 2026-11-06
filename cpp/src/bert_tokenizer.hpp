#pragma once

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace fireredpunc {

inline std::vector<uint32_t> Utf8Decode(const std::string& text) {
  std::vector<uint32_t> out;
  out.reserve(text.size());
  for (size_t i = 0; i < text.size();) {
    const unsigned char c = static_cast<unsigned char>(text[i]);
    uint32_t cp = 0;
    size_t n = 0;
    if (c < 0x80) {
      cp = c;
      n = 1;
    } else if ((c & 0xE0) == 0xC0) {
      cp = c & 0x1F;
      n = 2;
    } else if ((c & 0xF0) == 0xE0) {
      cp = c & 0x0F;
      n = 3;
    } else if ((c & 0xF8) == 0xF0) {
      cp = c & 0x07;
      n = 4;
    } else {
      out.push_back(0xFFFD);
      ++i;
      continue;
    }
    if (i + n > text.size()) {
      out.push_back(0xFFFD);
      break;
    }
    bool ok = true;
    for (size_t k = 1; k < n; ++k) {
      const unsigned char cc = static_cast<unsigned char>(text[i + k]);
      if ((cc & 0xC0) != 0x80) {
        ok = false;
        break;
      }
      cp = (cp << 6) | (cc & 0x3F);
    }
    if (!ok) {
      out.push_back(0xFFFD);
      ++i;
    } else {
      out.push_back(cp);
      i += n;
    }
  }
  return out;
}

inline std::string Utf8Encode(uint32_t cp) {
  std::string out;
  if (cp < 0x80) {
    out.push_back(static_cast<char>(cp));
  } else if (cp < 0x800) {
    out.push_back(static_cast<char>(0xC0 | (cp >> 6)));
    out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
  } else if (cp < 0x10000) {
    out.push_back(static_cast<char>(0xE0 | (cp >> 12)));
    out.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
    out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
  } else {
    out.push_back(static_cast<char>(0xF0 | (cp >> 18)));
    out.push_back(static_cast<char>(0x80 | ((cp >> 12) & 0x3F)));
    out.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
    out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
  }
  return out;
}

inline std::string Utf8FromCps(const std::vector<uint32_t>& cps) {
  std::string out;
  for (uint32_t cp : cps) out += Utf8Encode(cp);
  return out;
}

inline bool IsAsciiSpace(uint32_t cp) {
  return cp == ' ' || cp == '\t' || cp == '\n' || cp == '\r' ||
         cp == '\v' || cp == '\f';
}

inline bool IsWhitespace(uint32_t cp) {
  if (IsAsciiSpace(cp)) return true;
  return cp == 0x00A0 || cp == 0x1680 || (cp >= 0x2000 && cp <= 0x200A) ||
         cp == 0x2028 || cp == 0x2029 || cp == 0x202F || cp == 0x205F ||
         cp == 0x3000;
}

inline bool IsControl(uint32_t cp) {
  if (cp == 0 || cp == 0xFFFD) return true;
  if (cp <= 0x1F || (cp >= 0x7F && cp <= 0x9F)) return true;
  return false;
}

inline bool IsChinese(uint32_t cp) {
  return (cp >= 0x4E00 && cp <= 0x9FFF) ||
         (cp >= 0x3400 && cp <= 0x4DBF) ||
         (cp >= 0x20000 && cp <= 0x2A6DF) ||
         (cp >= 0x2A700 && cp <= 0x2B73F) ||
         (cp >= 0x2B740 && cp <= 0x2B81F) ||
         (cp >= 0x2B820 && cp <= 0x2CEAF) ||
         (cp >= 0xF900 && cp <= 0xFAFF) ||
         (cp >= 0x2F800 && cp <= 0x2FA1F);
}

inline bool IsAsciiPunctuation(uint32_t cp) {
  return (cp >= 33 && cp <= 47) || (cp >= 58 && cp <= 64) ||
         (cp >= 91 && cp <= 96) || (cp >= 123 && cp <= 126);
}

inline bool IsPunctuation(uint32_t cp) {
  if (IsAsciiPunctuation(cp)) return true;
  switch (cp) {
    case 0x3001: case 0x3002: case 0x3008: case 0x3009:
    case 0x300A: case 0x300B: case 0x300C: case 0x300D:
    case 0x300E: case 0x300F: case 0x3010: case 0x3011:
    case 0x3014: case 0x3015: case 0x3016: case 0x3017:
    case 0x3018: case 0x3019: case 0x301A: case 0x301B:
    case 0x301D: case 0x301E: case 0x301F: case 0x3030:
    case 0x303D: case 0x30FB: case 0xFE10: case 0xFE11:
    case 0xFE12: case 0xFE13: case 0xFE14: case 0xFE15:
    case 0xFE16: case 0xFE17: case 0xFE18: case 0xFE19:
    case 0xFE30: case 0xFE31: case 0xFE32: case 0xFE33:
    case 0xFE34: case 0xFE35: case 0xFE36: case 0xFE37:
    case 0xFE38: case 0xFE39: case 0xFE3A: case 0xFE3B:
    case 0xFE3C: case 0xFE3D: case 0xFE3E: case 0xFE3F:
    case 0xFE40: case 0xFE41: case 0xFE42: case 0xFE43:
    case 0xFE44: case 0xFE45: case 0xFE46: case 0xFE47:
    case 0xFE48: case 0xFE49: case 0xFE4A: case 0xFE4B:
    case 0xFE4C: case 0xFE4D: case 0xFE4E: case 0xFE4F:
    case 0xFF01: case 0xFF02: case 0xFF03: case 0xFF04:
    case 0xFF05: case 0xFF06: case 0xFF07: case 0xFF08:
    case 0xFF09: case 0xFF0A: case 0xFF0B: case 0xFF0C:
    case 0xFF0D: case 0xFF0E: case 0xFF0F: case 0xFF1A:
    case 0xFF1B: case 0xFF1C: case 0xFF1D: case 0xFF1E:
    case 0xFF1F: case 0xFF20: case 0xFF3B: case 0xFF3C:
    case 0xFF3D: case 0xFF3E: case 0xFF3F: case 0xFF40:
    case 0xFF5B: case 0xFF5C: case 0xFF5D: case 0xFF5E:
    case 0xFF5F: case 0xFF60: case 0xFF61: case 0xFF62:
    case 0xFF63: case 0xFF64: case 0xFF65:
    case 0x2010: case 0x2011: case 0x2012: case 0x2013:
    case 0x2014: case 0x2015: case 0x2016: case 0x2017:
    case 0x2018: case 0x2019: case 0x201A: case 0x201B:
    case 0x201C: case 0x201D: case 0x201E: case 0x201F:
    case 0x2020: case 0x2021: case 0x2022: case 0x2023:
    case 0x2024: case 0x2025: case 0x2026: case 0x2027:
    case 0x2030: case 0x2031: case 0x2032: case 0x2033:
    case 0x2034: case 0x2035: case 0x2036: case 0x2037:
    case 0x2038: case 0x2039: case 0x203A: case 0x203B:
    case 0x203C: case 0x203D: case 0x203E: case 0x203F:
    case 0x2040: case 0x2041: case 0x2042: case 0x2043:
    case 0x2044: case 0x2045: case 0x2046: case 0x2047:
    case 0x2048: case 0x2049: case 0x204A: case 0x204B:
    case 0x204C: case 0x204D: case 0x204E: case 0x204F:
    case 0x2050: case 0x2051: case 0x2052: case 0x2053:
    case 0x2054: case 0x2055: case 0x2056: case 0x2057:
    case 0x2058: case 0x2059: case 0x205A: case 0x205B:
    case 0x205C: case 0x205D: case 0x205E:
      return true;
    default:
      return false;
  }
}

inline uint32_t StripAccent(uint32_t cp) {
  if (cp >= 0x0300 && cp <= 0x036F) return 0;
  switch (cp) {
    case 0x00C0: case 0x00C1: case 0x00C2: case 0x00C3: case 0x00C4:
    case 0x00C5: return 'a';
    case 0x00E0: case 0x00E1: case 0x00E2: case 0x00E3: case 0x00E4:
    case 0x00E5: return 'a';
    case 0x00C7: return 'c'; case 0x00E7: return 'c';
    case 0x00C8: case 0x00C9: case 0x00CA: case 0x00CB: return 'e';
    case 0x00E8: case 0x00E9: case 0x00EA: case 0x00EB: return 'e';
    case 0x00CC: case 0x00CD: case 0x00CE: case 0x00CF: return 'i';
    case 0x00EC: case 0x00ED: case 0x00EE: case 0x00EF: return 'i';
    case 0x00D1: return 'n'; case 0x00F1: return 'n';
    case 0x00D2: case 0x00D3: case 0x00D4: case 0x00D5: case 0x00D6:
      return 'o';
    case 0x00F2: case 0x00F3: case 0x00F4: case 0x00F5: case 0x00F6:
      return 'o';
    case 0x00D9: case 0x00DA: case 0x00DB: case 0x00DC: return 'u';
    case 0x00F9: case 0x00FA: case 0x00FB: case 0x00FC: return 'u';
    case 0x00DD: return 'y'; case 0x00FD: case 0x00FF: return 'y';
    case 0x0100: case 0x0102: case 0x0104: return 'a';
    case 0x0101: case 0x0103: case 0x0105: return 'a';
    case 0x0106: case 0x0108: case 0x010A: case 0x010C: return 'c';
    case 0x0107: case 0x0109: case 0x010B: case 0x010D: return 'c';
    case 0x010E: return 'd'; case 0x010F: return 'd';
    case 0x0112: case 0x0114: case 0x0116: case 0x0118: return 'e';
    case 0x0113: case 0x0115: case 0x0117: case 0x0119: return 'e';
    case 0x011A: case 0x011C: case 0x011E: case 0x0120: case 0x0122:
      return 'g';
    case 0x011B: case 0x011D: case 0x011F: case 0x0121: case 0x0123:
      return 'g';
    case 0x0124: return 'h'; case 0x0125: return 'h';
    case 0x0128: case 0x012A: case 0x012C: case 0x012E: case 0x0130:
      return 'i';
    case 0x0129: case 0x012B: case 0x012D: case 0x012F: case 0x0131:
      return 'i';
    case 0x0134: return 'j'; case 0x0135: return 'j';
    case 0x0136: return 'k'; case 0x0137: return 'k';
    case 0x0139: case 0x013B: case 0x013D: case 0x013F: return 'l';
    case 0x013A: case 0x013C: case 0x013E: case 0x0140: return 'l';
    case 0x0143: case 0x0145: case 0x0147: return 'n';
    case 0x0144: case 0x0146: case 0x0148: return 'n';
    case 0x014C: case 0x014E: case 0x0150: return 'o';
    case 0x014D: case 0x014F: case 0x0151: return 'o';
    case 0x0154: case 0x0156: case 0x0158: return 'r';
    case 0x0155: case 0x0157: case 0x0159: return 'r';
    case 0x015A: case 0x015C: case 0x015E: case 0x0160: return 's';
    case 0x015B: case 0x015D: case 0x015F: case 0x0161: return 's';
    case 0x0162: case 0x0164: return 't'; case 0x0163: case 0x0165: return 't';
    case 0x0168: case 0x016A: case 0x016C: case 0x016E: case 0x0170:
    case 0x0172: return 'u';
    case 0x0169: case 0x016B: case 0x016D: case 0x016F: case 0x0171:
    case 0x0173: return 'u';
    case 0x0174: return 'w'; case 0x0175: return 'w';
    case 0x0176: return 'y'; case 0x0177: return 'y';
    case 0x0178: return 'y'; case 0x0179: case 0x017B: case 0x017D: return 'z';
    case 0x017A: case 0x017C: case 0x017E: return 'z';
    default: return cp;
  }
}

inline char AsciiLower(char c) {
  return (c >= 'A' && c <= 'Z') ? static_cast<char>(c - 'A' + 'a') : c;
}

inline char AsciiUpper(char c) {
  return (c >= 'a' && c <= 'z') ? static_cast<char>(c - 'a' + 'A') : c;
}

inline std::string AsciiLowerCopy(const std::string& s) {
  std::string out = s;
  for (char& c : out) c = AsciiLower(c);
  return out;
}

inline std::string ReplaceAll(std::string s, const std::string& from,
                              const std::string& to) {
  if (from.empty()) return s;
  size_t pos = 0;
  while ((pos = s.find(from, pos)) != std::string::npos) {
    s.replace(pos, from.size(), to);
    pos += to.size();
  }
  return s;
}

inline std::string Trim(const std::string& s) {
  size_t b = 0, e = s.size();
  while (b < e && IsAsciiSpace(static_cast<unsigned char>(s[b]))) ++b;
  while (e > b && IsAsciiSpace(static_cast<unsigned char>(s[e - 1]))) --e;
  return s.substr(b, e - b);
}

inline std::string PrefixCodepoints(const std::string& text, size_t n) {
  const auto cps = Utf8Decode(text);
  if (n >= cps.size()) return text;
  std::vector<uint32_t> head(cps.begin(), cps.begin() + n);
  return Utf8FromCps(head);
}

class BertTokenizer {
 public:
  bool Load(const std::string& vocab_path) {
    std::ifstream in(vocab_path);
    if (!in) return false;
    std::string line;
    int index = 0;
    while (std::getline(in, line)) {
      if (!line.empty() && line.back() == '\r') line.pop_back();
      std::string token = line;
      if (token == "<space>") token = " ";
      vocab_[token] = index;
      id2word_.push_back(token);
      ++index;
    }
    auto it = vocab_.find("[UNK]");
    unk_id_ = it == vocab_.end() ? 0 : it->second;
    return !vocab_.empty();
  }

  const std::unordered_map<std::string, int>& vocab() const { return vocab_; }
  int unk_id() const { return unk_id_; }

  std::vector<std::string> Tokenize(const std::string& text,
                                    bool recover_unk = true) const {
    std::vector<std::string> tokens;
    for (const std::string& word : BasicTokenize(text)) {
      const auto pieces = WordPiece(word);
      tokens.insert(tokens.end(), pieces.begin(), pieces.end());
    }
    if (recover_unk) tokens = RecoverUnk(AsciiLowerCopy(text), tokens);
    return tokens;
  }

  std::vector<int> ToIds(const std::vector<std::string>& tokens) const {
    std::vector<int> ids;
    ids.reserve(tokens.size());
    for (const std::string& token : tokens) {
      auto it = vocab_.find(token);
      ids.push_back(it == vocab_.end() ? unk_id_ : it->second);
    }
    return ids;
  }

 private:
  std::vector<std::string> BasicTokenize(const std::string& text) const {
    std::string cleaned;
    for (uint32_t cp : Utf8Decode(text)) {
      if (IsControl(cp)) continue;
      if (IsWhitespace(cp)) {
        cleaned.push_back(' ');
      } else {
        if (cp < 0x80) cp = static_cast<unsigned char>(AsciiLower(static_cast<char>(cp)));
        if (cp >= 0x0300 && cp <= 0x036F) continue;
        cp = StripAccent(cp);
        if (cp == 0) continue;
        cleaned += Utf8Encode(cp);
      }
    }

    std::vector<std::string> out;
    std::string current;
    auto flush = [&]() {
      if (!current.empty()) {
        out.push_back(current);
        current.clear();
      }
    };
    for (uint32_t cp : Utf8Decode(cleaned)) {
      if (cp == ' ' || IsWhitespace(cp)) {
        flush();
      } else if (IsChinese(cp) || IsPunctuation(cp)) {
        flush();
        out.push_back(Utf8Encode(cp));
      } else {
        current += Utf8Encode(cp);
      }
    }
    flush();
    return out;
  }

  std::vector<std::string> WordPiece(const std::string& word) const {
    const auto cps = Utf8Decode(word);
    if (cps.empty()) return {};
    if (cps.size() > 100) return {"[UNK]"};
    std::vector<std::string> pieces;
    size_t start = 0;
    while (start < cps.size()) {
      size_t end = cps.size();
      std::string match;
      while (start < end) {
        std::vector<uint32_t> sub(cps.begin() + start, cps.begin() + end);
        std::string piece = (start > 0 ? "##" : "") + Utf8FromCps(sub);
        if (vocab_.find(piece) != vocab_.end()) {
          match = piece;
          break;
        }
        --end;
      }
      if (match.empty()) return {"[UNK]"};
      pieces.push_back(match);
      start = end;
    }
    return pieces;
  }

  std::vector<std::string> RecoverUnk(const std::string& text_lower,
                                      const std::vector<std::string>& tokens) const {
    bool has_unk = false;
    for (const auto& token : tokens) {
      if (token == "[UNK]") {
        has_unk = true;
        break;
      }
    }
    if (!has_unk) return tokens;

    std::string text_no_space;
    for (char c : text_lower) {
      if (c != ' ') text_no_space.push_back(c);
    }

    bool all_non_alnum = true;
    for (uint32_t cp : Utf8Decode(text_lower)) {
      if ((cp < 0x80 && std::isalnum(static_cast<unsigned char>(cp))) ||
          cp == '\'') {
        all_non_alnum = false;
        break;
      }
    }
    if (all_non_alnum) {
      const auto cps = Utf8Decode(text_no_space);
      if (cps.size() == tokens.size()) {
        bool success = true;
        std::vector<std::string> recovered;
        for (size_t i = 0; i < cps.size(); ++i) {
          const std::string ch = Utf8Encode(cps[i]);
          if (tokens[i] != "[UNK]" && tokens[i] != ch) {
            success = false;
            break;
          }
          recovered.push_back(ch);
        }
        if (success) return recovered;
      }
    }

    std::vector<std::string> recovered;
    size_t text_pos = 0;
    for (size_t i = 0; i < tokens.size();) {
      if (tokens[i] != "[UNK]") {
        recovered.push_back(tokens[i]);
        text_pos = std::min(text_no_space.size(),
                            text_pos + ReplaceAll(tokens[i], "##", "").size());
        ++i;
        continue;
      }
      size_t j = i;
      while (j < tokens.size() && tokens[j] == "[UNK]") ++j;
      const size_t unk_count = j - i;
      std::string post_token;
      if (j < tokens.size()) post_token = ReplaceAll(tokens[j], "##", "");

      std::string remaining = text_pos < text_no_space.size()
                                  ? text_no_space.substr(text_pos)
                                  : std::string();
      std::string unk_chars;
      if (!post_token.empty()) {
        const size_t anchor = remaining.find(post_token);
        if (anchor != std::string::npos) {
          unk_chars = remaining.substr(0, anchor);
        } else {
          unk_chars = PrefixCodepoints(remaining, unk_count);
        }
      } else {
        unk_chars = PrefixCodepoints(remaining, unk_count);
      }
      const auto chars = Utf8Decode(unk_chars);
      for (size_t k = 0; k < unk_count; ++k) {
        recovered.push_back(k < chars.size() ? Utf8Encode(chars[k]) : "");
      }
      text_pos = std::min(text_no_space.size(), text_pos + unk_chars.size());
      i = j;
    }

    std::vector<std::string> filtered;
    for (const auto& token : recovered) {
      if (!token.empty() && token != "[UNK]") filtered.push_back(token);
    }
    return filtered;
  }

  std::unordered_map<std::string, int> vocab_;
  std::vector<std::string> id2word_;
  int unk_id_ = 0;
};

class TokenDict {
 public:
  bool Load(const std::string& path, const std::string& unk = "") {
    std::ifstream in(path);
    if (!in) return false;
    std::string line;
    int index = 0;
    while (std::getline(in, line)) {
      if (!line.empty() && line.back() == '\r') line.pop_back();
      std::string word = line;
      int id = index;
      const size_t space = line.find_last_of(" \t");
      if (space != std::string::npos) {
        const std::string tail = line.substr(space + 1);
        char* end = nullptr;
        const long parsed = std::strtol(tail.c_str(), &end, 10);
        if (end && *end == '\0' && !tail.empty()) {
          word = line.substr(0, space);
          id = static_cast<int>(parsed);
        }
      }
      if (word == "<space>") word = " ";
      id2word_[id] = word;
      word2id_[word] = id;
      if (id >= static_cast<int>(max_id_)) max_id_ = id;
      ++index;
    }
    auto it = word2id_.find(unk);
    if (!unk.empty() && it == word2id_.end()) return false;
    unk_id_ = it == word2id_.end() ? -1 : it->second;
    return !word2id_.empty();
  }

  int Get(const std::string& key, int fallback) const {
    auto it = word2id_.find(key);
    return it == word2id_.end() ? fallback : it->second;
  }

  std::string Word(int id) const {
    auto it = id2word_.find(id);
    return it == id2word_.end() ? std::string() : it->second;
  }

  int max_id() const { return max_id_; }

 private:
  std::unordered_map<std::string, int> word2id_;
  std::unordered_map<int, std::string> id2word_;
  int unk_id_ = -1;
  int max_id_ = -1;
};

}  // namespace fireredpunc
