// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (c) 2026 Guangzhou Songzi Intelligent Technology Services Co., Ltd.

/* Strict RFC 8259 JSON reader and the wire writer; rules in json.hpp. */
#include "util/json.hpp"

#include <cstdio>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <xlocale.h>

namespace spk::json {

// ---- Value ------------------------------------------------------------------------------------

namespace {
[[noreturn]] void type_error(const char* want) {
    throw std::logic_error(std::string("json value is not ") + want);
}
}  // namespace

bool Value::as_bool() const {
    if (auto p = std::get_if<bool>(&v_)) return *p;
    type_error("a bool");
}
double Value::as_number() const {
    if (auto p = std::get_if<double>(&v_)) return *p;
    type_error("a number");
}
const std::string& Value::as_string() const {
    if (auto p = std::get_if<std::string>(&v_)) return *p;
    type_error("a string");
}
const Array& Value::as_array() const {
    if (auto p = std::get_if<Array>(&v_)) return *p;
    type_error("an array");
}
Array& Value::as_array() {
    if (auto p = std::get_if<Array>(&v_)) return *p;
    type_error("an array");
}
const Object& Value::as_object() const {
    if (auto p = std::get_if<Object>(&v_)) return *p;
    type_error("an object");
}
Object& Value::as_object() {
    if (auto p = std::get_if<Object>(&v_)) return *p;
    type_error("an object");
}

const Value* Value::find(std::string_view key) const {
    auto o = std::get_if<Object>(&v_);
    if (!o) return nullptr;
    auto it = o->find(key);
    return it == o->end() ? nullptr : &it->second;
}

Value& Value::operator[](std::string_view key) {
    if (is_null()) v_ = Object{};
    Object& o = as_object();
    auto it = o.find(key);
    if (it == o.end()) it = o.emplace(std::string(key), Value{}).first;
    return it->second;
}

void Value::push_back(Value v) {
    if (is_null()) v_ = Array{};
    as_array().push_back(std::move(v));
}

ParseError::ParseError(size_t offset, const std::string& what)
    : std::runtime_error("JSON parse error at byte " + std::to_string(offset) + ": " + what),
      offset_(offset) {}

// ---- reader -----------------------------------------------------------------------------------

namespace {

locale_t c_locale() {
    static locale_t loc = newlocale(LC_ALL_MASK, "C", nullptr);
    return loc;
}

/* Parser: recursive descent over the UTF-8 text
 *   run(): skip whitespace; value(0); skip whitespace; anything left is an error
 *   value(depth): by first byte: object, array, string, true, false, null, number
 *   object, array: depth + 1 > 64 is an error
 *   string: \uXXXX surrogate pairs combined, unpaired refused; raw bytes >= 0x80 validated as UTF-8
 *   number: validate the grammar first, then strtod_l in the C locale; infinity refused
 */
class Parser {
public:
    explicit Parser(std::string_view t) : s_(t) {}

    Value run() {
        skip_ws();
        Value v = value(0);
        skip_ws();
        if (i_ != s_.size()) err("unexpected data after the top-level value");
        return v;
    }

private:
    std::string_view s_;
    size_t i_ = 0;

    [[noreturn]] void err(const std::string& what) const { throw ParseError(i_, what); }
    [[noreturn]] void err_at(size_t at, const std::string& what) const { throw ParseError(at, what); }

    bool eof() const { return i_ >= s_.size(); }
    unsigned char peek() const { return static_cast<unsigned char>(s_[i_]); }

    void skip_ws() {
        while (!eof()) {
            char c = s_[i_];
            if (c == ' ' || c == '\t' || c == '\n' || c == '\r') ++i_;
            else break;
        }
    }

    void expect_literal(std::string_view lit) {
        if (s_.substr(i_, lit.size()) != lit) err("invalid literal");
        i_ += lit.size();
    }

    Value value(int depth) {
        if (eof()) err("unexpected end of input");
        unsigned char c = peek();
        switch (c) {
            case '{': return object(depth + 1);
            case '[': return array(depth + 1);
            case '"': return Value(string());
            case 't': expect_literal("true"); return Value(true);
            case 'f': expect_literal("false"); return Value(false);
            case 'n': expect_literal("null"); return Value(nullptr);
            default:
                if (c == '-' || (c >= '0' && c <= '9')) return Value(number());
                err("unexpected character");
        }
    }

    Value object(int depth) {
        if (depth > kMaxDepth) err("nesting deeper than 64");
        ++i_;  // '{'
        Object o;
        skip_ws();
        if (!eof() && s_[i_] == '}') { ++i_; return Value(std::move(o)); }
        for (;;) {
            skip_ws();
            if (eof() || s_[i_] != '"') err("expected a string key");
            std::string k = string();
            skip_ws();
            if (eof() || s_[i_] != ':') err("expected ':'");
            ++i_;
            skip_ws();
            Value v = value(depth);
            o.insert_or_assign(std::move(k), std::move(v));  // last duplicate wins
            skip_ws();
            if (eof()) err("unexpected end of input in an object");
            if (s_[i_] == ',') { ++i_; continue; }
            if (s_[i_] == '}') { ++i_; break; }
            err("expected ',' or '}'");
        }
        return Value(std::move(o));
    }

    Value array(int depth) {
        if (depth > kMaxDepth) err("nesting deeper than 64");
        ++i_;  // '['
        Array a;
        skip_ws();
        if (!eof() && s_[i_] == ']') { ++i_; return Value(std::move(a)); }
        for (;;) {
            skip_ws();
            a.push_back(value(depth));
            skip_ws();
            if (eof()) err("unexpected end of input in an array");
            if (s_[i_] == ',') { ++i_; continue; }
            if (s_[i_] == ']') { ++i_; break; }
            err("expected ',' or ']'");
        }
        return Value(std::move(a));
    }

    static void put_utf8(std::string& out, uint32_t cp) {
        if (cp < 0x80) {
            out += static_cast<char>(cp);
        } else if (cp < 0x800) {
            out += static_cast<char>(0xC0 | (cp >> 6));
            out += static_cast<char>(0x80 | (cp & 0x3F));
        } else if (cp < 0x10000) {
            out += static_cast<char>(0xE0 | (cp >> 12));
            out += static_cast<char>(0x80 | ((cp >> 6) & 0x3F));
            out += static_cast<char>(0x80 | (cp & 0x3F));
        } else {
            out += static_cast<char>(0xF0 | (cp >> 18));
            out += static_cast<char>(0x80 | ((cp >> 12) & 0x3F));
            out += static_cast<char>(0x80 | ((cp >> 6) & 0x3F));
            out += static_cast<char>(0x80 | (cp & 0x3F));
        }
    }

    uint32_t hex4() {
        if (i_ + 4 > s_.size()) err("truncated \\u escape");
        uint32_t v = 0;
        for (int k = 0; k < 4; ++k) {
            char c = s_[i_++];
            v <<= 4;
            if (c >= '0' && c <= '9') v |= uint32_t(c - '0');
            else if (c >= 'a' && c <= 'f') v |= uint32_t(c - 'a' + 10);
            else if (c >= 'A' && c <= 'F') v |= uint32_t(c - 'A' + 10);
            else err_at(i_ - 1, "invalid hex digit in \\u escape");
        }
        return v;
    }

    // Validates one UTF-8 sequence starting at i_ (first byte >= 0x80) and copies it.
    void utf8_sequence(std::string& out) {
        size_t start = i_;
        unsigned char c = peek();
        int n;
        uint32_t cp, min;
        if (c >= 0xC2 && c <= 0xDF) { n = 1; cp = c & 0x1F; min = 0x80; }
        else if (c >= 0xE0 && c <= 0xEF) { n = 2; cp = c & 0x0F; min = 0x800; }
        else if (c >= 0xF0 && c <= 0xF4) { n = 3; cp = c & 0x07; min = 0x10000; }
        else err("invalid UTF-8");
        if (i_ + 1 + size_t(n) > s_.size()) err("truncated UTF-8 sequence");
        for (int k = 1; k <= n; ++k) {
            unsigned char cc = static_cast<unsigned char>(s_[i_ + size_t(k)]);
            if ((cc & 0xC0) != 0x80) err("invalid UTF-8");
            cp = (cp << 6) | (cc & 0x3F);
        }
        if (cp < min || cp > 0x10FFFF || (cp >= 0xD800 && cp <= 0xDFFF)) err("invalid UTF-8");
        out.append(s_.substr(start, size_t(n) + 1));
        i_ += size_t(n) + 1;
    }

    std::string string() {
        ++i_;  // opening quote
        std::string out;
        for (;;) {
            if (eof()) err("unterminated string");
            unsigned char c = peek();
            if (c == '"') { ++i_; return out; }
            if (c < 0x20) err("unescaped control character in a string");
            if (c == '\\') {
                ++i_;
                if (eof()) err("unterminated escape");
                char e = s_[i_++];
                switch (e) {
                    case '"': out += '"'; break;
                    case '\\': out += '\\'; break;
                    case '/': out += '/'; break;
                    case 'b': out += '\b'; break;
                    case 'f': out += '\f'; break;
                    case 'n': out += '\n'; break;
                    case 'r': out += '\r'; break;
                    case 't': out += '\t'; break;
                    case 'u': {
                        size_t at = i_ - 2;
                        uint32_t cp = hex4();
                        if (cp >= 0xD800 && cp <= 0xDBFF) {
                            if (i_ + 2 > s_.size() || s_[i_] != '\\' || s_[i_ + 1] != 'u')
                                err_at(at, "unpaired surrogate in \\u escape");
                            i_ += 2;
                            uint32_t lo = hex4();
                            if (lo < 0xDC00 || lo > 0xDFFF) err_at(at, "unpaired surrogate in \\u escape");
                            cp = 0x10000 + ((cp - 0xD800) << 10) + (lo - 0xDC00);
                        } else if (cp >= 0xDC00 && cp <= 0xDFFF) {
                            err_at(at, "unpaired surrogate in \\u escape");
                        }
                        put_utf8(out, cp);
                        break;
                    }
                    default: err_at(i_ - 1, "invalid escape");
                }
                continue;
            }
            if (c >= 0x80) { utf8_sequence(out); continue; }
            out += static_cast<char>(c);
            ++i_;
        }
    }

    double number() {
        size_t start = i_;
        if (s_[i_] == '-') ++i_;
        if (eof()) err("invalid number");
        if (s_[i_] == '0') {
            ++i_;
        } else if (s_[i_] >= '1' && s_[i_] <= '9') {
            while (!eof() && s_[i_] >= '0' && s_[i_] <= '9') ++i_;
        } else {
            err("invalid number");
        }
        if (!eof() && s_[i_] == '.') {
            ++i_;
            if (eof() || s_[i_] < '0' || s_[i_] > '9') err("invalid number: digit expected after '.'");
            while (!eof() && s_[i_] >= '0' && s_[i_] <= '9') ++i_;
        }
        if (!eof() && (s_[i_] == 'e' || s_[i_] == 'E')) {
            ++i_;
            if (!eof() && (s_[i_] == '+' || s_[i_] == '-')) ++i_;
            if (eof() || s_[i_] < '0' || s_[i_] > '9') err("invalid number: digit expected in exponent");
            while (!eof() && s_[i_] >= '0' && s_[i_] <= '9') ++i_;
        }
        // The literal has been validated against the RFC 8259 grammar above; strtod_l in the C
        // locale gives the correctly rounded double (std::from_chars for double needs macOS 26).
        std::string tmp(s_.data() + start, i_ - start);
        char* end = nullptr;
        double v = strtod_l(tmp.c_str(), &end, c_locale());
        if (end != tmp.c_str() + tmp.size()) err_at(start, "invalid number");
        if (std::isinf(v)) err_at(start, "number out of range");
        return v;
    }
};

}  // namespace

Value parse(std::string_view text) { return Parser(text).run(); }

// ---- writer -----------------------------------------------------------------------------------

void append_number(std::string& out, double d) {
    if (!std::isfinite(d)) { out += "null"; return; }
    char buf[64];
    int n = snprintf_l(buf, sizeof buf, c_locale(), "%.17g", d);
    out.append(buf, size_t(n));
}

std::string format_number(double d) {
    std::string s;
    append_number(s, d);
    return s;
}

void append_string(std::string& out, std::string_view s) {
    static const char hex[] = "0123456789abcdef";
    out += '"';
    for (char ch : s) {
        unsigned char c = static_cast<unsigned char>(ch);
        switch (c) {
            case '"': out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\n': out += "\\n"; break;
            case '\r': out += "\\r"; break;
            case '\t': out += "\\t"; break;
            case '\b': out += "\\b"; break;
            case '\f': out += "\\f"; break;
            default:
                if (c < 0x20) {
                    out += "\\u00";
                    out += hex[c >> 4];
                    out += hex[c & 15];
                } else {
                    out += ch;
                }
        }
    }
    out += '"';
}

void write_to(std::string& out, const Value& v) {
    switch (v.type()) {
        case Type::Null: out += "null"; break;
        case Type::Bool: out += v.as_bool() ? "true" : "false"; break;
        case Type::Number: append_number(out, v.as_number()); break;
        case Type::String: append_string(out, v.as_string()); break;
        case Type::Array: {
            out += '[';
            bool first = true;
            for (const Value& e : v.as_array()) {
                if (!first) out += ',';
                first = false;
                write_to(out, e);
            }
            out += ']';
            break;
        }
        case Type::Object: {
            out += '{';
            bool first = true;
            for (const auto& [k, e] : v.as_object()) {
                if (!first) out += ',';
                first = false;
                append_string(out, k);
                out += ':';
                write_to(out, e);
            }
            out += '}';
            break;
        }
    }
}

std::string write(const Value& v) {
    std::string s;
    write_to(s, v);
    return s;
}

}  // namespace spk::json
