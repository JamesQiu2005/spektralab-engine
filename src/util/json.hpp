// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (c) 2026 Guangzhou Songzi Intelligent Technology Services Co., Ltd.

/* Strict RFC 8259 JSON reader and the wire writer.
 *
 * reader
 *   UTF-8 validated
 *   numbers: JSON number literals only; no NaN, Infinity or hex
 *            overflow of a double: parse error; underflow: correctly rounded, may be 0
 *   duplicate keys: last one wins
 *   nesting: depth <= 64; a top-level array or object is depth 1
 *   after the top-level value: whitespace only
 *   errors carry the byte offset
 *
 * writer
 *   compact; object keys in ascending bytewise order
 *   numbers: printf "%.17g", so integers print without a fraction; non-finite -> null
 *   strings: escape '"', '\\', \n \r \t \b \f; other C0 controls as \u00xx in lower-case hex;
 *            all other bytes unchanged
 */
#pragma once

#include <cstddef>
#include <cstdint>
#include <map>
#include <stdexcept>
#include <string>
#include <string_view>
#include <variant>
#include <vector>

namespace spk::json {

class Value;
using Array = std::vector<Value>;
using Object = std::map<std::string, Value, std::less<>>;

enum class Type : uint8_t { Null, Bool, Number, String, Array, Object };

class Value {
public:
    Value() = default;
    Value(std::nullptr_t) {}
    Value(bool b) : v_(b) {}
    Value(double d) : v_(d) {}
    Value(float d) : v_(static_cast<double>(d)) {}
    Value(int i) : v_(static_cast<double>(i)) {}
    Value(long i) : v_(static_cast<double>(i)) {}
    Value(long long i) : v_(static_cast<double>(i)) {}
    Value(unsigned i) : v_(static_cast<double>(i)) {}
    Value(unsigned long i) : v_(static_cast<double>(i)) {}
    Value(unsigned long long i) : v_(static_cast<double>(i)) {}
    Value(const char* s) : v_(std::string(s)) {}
    Value(std::string s) : v_(std::move(s)) {}
    Value(std::string_view s) : v_(std::string(s)) {}
    Value(Array a) : v_(std::move(a)) {}
    Value(Object o) : v_(std::move(o)) {}

    Type type() const noexcept { return static_cast<Type>(v_.index()); }
    bool is_null() const noexcept { return type() == Type::Null; }
    bool is_bool() const noexcept { return type() == Type::Bool; }
    bool is_number() const noexcept { return type() == Type::Number; }
    bool is_string() const noexcept { return type() == Type::String; }
    bool is_array() const noexcept { return type() == Type::Array; }
    bool is_object() const noexcept { return type() == Type::Object; }

    // Accessors throw std::logic_error on a type mismatch (a bug, not a user error).
    bool as_bool() const;
    double as_number() const;
    const std::string& as_string() const;
    const Array& as_array() const;
    Array& as_array();
    const Object& as_object() const;
    Object& as_object();

    // Object helpers. find() returns nullptr when this is not an object or the key is absent.
    const Value* find(std::string_view key) const;
    // Turns a null value into an empty object first; throws std::logic_error on another type.
    Value& operator[](std::string_view key);
    // Turns a null value into an empty array first.
    void push_back(Value v);

    bool operator==(const Value& o) const { return v_ == o.v_; }

private:
    std::variant<std::monostate, bool, double, std::string, Array, Object> v_;
};

class ParseError : public std::runtime_error {
public:
    ParseError(size_t offset, const std::string& what);
    size_t offset() const noexcept { return offset_; }

private:
    size_t offset_;
};

constexpr int kMaxDepth = 64;

// Throws ParseError with message "JSON parse error at byte <offset>: <reason>".
Value parse(std::string_view text);

// Compact wire form.
std::string write(const Value& v);
void write_to(std::string& out, const Value& v);

// "%.17g" in the C locale; non-finite -> "null".
std::string format_number(double d);
void append_number(std::string& out, double d);
// Appends "\"...\"" with the escaping rules above.
void append_string(std::string& out, std::string_view s);

}  // namespace spk::json
