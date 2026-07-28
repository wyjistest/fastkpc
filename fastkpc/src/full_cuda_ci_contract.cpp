#include "full_cuda_ci_contract.hpp"

#include <algorithm>
#include <array>
#include <charconv>
#include <cstdint>
#include <iomanip>
#include <limits>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace fastkpc {
namespace {

constexpr std::int64_t kMaxSafeJsonInteger = 9007199254740992LL;

enum class JsonType {
  Null,
  Boolean,
  Integer,
  String,
  Array,
  Object
};

struct JsonValue {
  JsonType type = JsonType::Null;
  bool boolean = false;
  std::int64_t integer = 0;
  std::string string;
  std::vector<JsonValue> array;
  std::map<std::string, JsonValue> object;

  static JsonValue null_value() {
    return JsonValue{};
  }

  static JsonValue boolean_value(bool value) {
    JsonValue output;
    output.type = JsonType::Boolean;
    output.boolean = value;
    return output;
  }

  static JsonValue integer_value(std::int64_t value) {
    JsonValue output;
    output.type = JsonType::Integer;
    output.integer = value;
    return output;
  }

  static JsonValue string_value(std::string value) {
    JsonValue output;
    output.type = JsonType::String;
    output.string = std::move(value);
    return output;
  }

  static JsonValue array_value(std::vector<JsonValue> value) {
    JsonValue output;
    output.type = JsonType::Array;
    output.array = std::move(value);
    return output;
  }

  static JsonValue object_value(std::map<std::string, JsonValue> value) {
    JsonValue output;
    output.type = JsonType::Object;
    output.object = std::move(value);
    return output;
  }
};

class JsonParser {
 public:
  explicit JsonParser(const std::string& input) : input_(input) {}

  JsonValue parse() {
    skip_whitespace();
    JsonValue value = parse_value();
    skip_whitespace();
    if (position_ != input_.size()) fail("trailing content after JSON value");
    return value;
  }

 private:
  const std::string& input_;
  std::size_t position_ = 0;

  [[noreturn]] void fail(const std::string& message) const {
    throw std::runtime_error(
      "tracked contract JSON parse error at byte " +
      std::to_string(position_) + ": " + message);
  }

  bool at_end() const {
    return position_ >= input_.size();
  }

  char peek() const {
    return at_end() ? '\0' : input_[position_];
  }

  char take() {
    if (at_end()) fail("unexpected end of input");
    return input_[position_++];
  }

  void skip_whitespace() {
    while (!at_end()) {
      const char value = peek();
      if (value != ' ' && value != '\t' && value != '\n' && value != '\r') {
        break;
      }
      ++position_;
    }
  }

  JsonValue parse_value() {
    if (at_end()) fail("expected JSON value");
    switch (peek()) {
    case 'n':
      parse_literal("null");
      return JsonValue::null_value();
    case 't':
      parse_literal("true");
      return JsonValue::boolean_value(true);
    case 'f':
      parse_literal("false");
      return JsonValue::boolean_value(false);
    case '"':
      return JsonValue::string_value(parse_string());
    case '[':
      return parse_array();
    case '{':
      return parse_object();
    default:
      if (peek() == '-' || (peek() >= '0' && peek() <= '9')) {
        return parse_integer();
      }
      fail("expected JSON value");
    }
  }

  void parse_literal(const char* literal) {
    for (const char* cursor = literal; *cursor != '\0'; ++cursor) {
      if (take() != *cursor) fail("invalid JSON literal");
    }
  }

  int parse_hex_digit() {
    const char value = take();
    if (value >= '0' && value <= '9') return value - '0';
    if (value >= 'a' && value <= 'f') return value - 'a' + 10;
    if (value >= 'A' && value <= 'F') return value - 'A' + 10;
    fail("invalid Unicode escape");
  }

  unsigned int parse_hex_quad() {
    unsigned int output = 0;
    for (int index = 0; index < 4; ++index) {
      output = (output << 4U) | static_cast<unsigned int>(parse_hex_digit());
    }
    return output;
  }

  std::string parse_string() {
    if (take() != '"') fail("expected string");
    std::string output;
    while (true) {
      if (at_end()) fail("unterminated string");
      const unsigned char value = static_cast<unsigned char>(take());
      if (value == '"') break;
      if (value < 0x20U) fail("unescaped control character in string");
      if (value >= 0x80U) {
        fail("tracked contract strings must be ASCII");
      }
      if (value != '\\') {
        output.push_back(static_cast<char>(value));
        continue;
      }
      const char escape = take();
      switch (escape) {
      case '"': output.push_back('"'); break;
      case '\\': output.push_back('\\'); break;
      case '/': output.push_back('/'); break;
      case 'b': output.push_back('\b'); break;
      case 'f': output.push_back('\f'); break;
      case 'n': output.push_back('\n'); break;
      case 'r': output.push_back('\r'); break;
      case 't': output.push_back('\t'); break;
      case 'u': {
        const unsigned int codepoint = parse_hex_quad();
        if (codepoint >= 0xd800U && codepoint <= 0xdfffU) {
          fail("surrogate escapes are not allowed in ASCII contracts");
        }
        if (codepoint > 0x7fU) {
          fail("tracked contract strings must be ASCII");
        }
        output.push_back(static_cast<char>(codepoint));
        break;
      }
      default:
        fail("invalid string escape");
      }
    }
    return output;
  }

  JsonValue parse_array() {
    take();
    skip_whitespace();
    std::vector<JsonValue> values;
    if (peek() == ']') {
      take();
      return JsonValue::array_value(std::move(values));
    }
    while (true) {
      skip_whitespace();
      values.push_back(parse_value());
      skip_whitespace();
      const char separator = take();
      if (separator == ']') break;
      if (separator != ',') fail("expected comma or closing array bracket");
      skip_whitespace();
    }
    return JsonValue::array_value(std::move(values));
  }

  JsonValue parse_object() {
    take();
    skip_whitespace();
    std::map<std::string, JsonValue> values;
    if (peek() == '}') {
      take();
      return JsonValue::object_value(std::move(values));
    }
    while (true) {
      skip_whitespace();
      if (peek() != '"') fail("object key must be a string");
      std::string key = parse_string();
      if (key.empty()) fail("object key must not be empty");
      skip_whitespace();
      if (take() != ':') fail("expected colon after object key");
      skip_whitespace();
      JsonValue value = parse_value();
      if (!values.emplace(key, std::move(value)).second) {
        fail("duplicate object key: " + key);
      }
      skip_whitespace();
      const char separator = take();
      if (separator == '}') break;
      if (separator != ',') fail("expected comma or closing object brace");
      skip_whitespace();
    }
    return JsonValue::object_value(std::move(values));
  }

  JsonValue parse_integer() {
    const std::size_t start = position_;
    if (peek() == '-') ++position_;
    if (at_end()) fail("invalid JSON number");
    if (peek() == '0') {
      ++position_;
      if (!at_end() && peek() >= '0' && peek() <= '9') {
        fail("leading zero in JSON number");
      }
    } else {
      if (peek() < '1' || peek() > '9') fail("invalid JSON number");
      while (!at_end() && peek() >= '0' && peek() <= '9') ++position_;
    }
    if (!at_end() && (peek() == '.' || peek() == 'e' || peek() == 'E')) {
      while (!at_end()) {
        const char value = peek();
        if ((value >= '0' && value <= '9') || value == '.' || value == 'e' ||
            value == 'E' || value == '+' || value == '-') {
          ++position_;
        } else {
          break;
        }
      }
      fail("tracked contracts require integer JSON numbers; finite decimals must be strings");
    }
    const std::string token = input_.substr(start, position_ - start);
    std::int64_t value = 0;
    const auto parsed = std::from_chars(
      token.data(), token.data() + token.size(), value, 10);
    if (parsed.ec != std::errc() || parsed.ptr != token.data() + token.size()) {
      fail("JSON integer is outside the supported range");
    }
    if (value < -kMaxSafeJsonInteger || value > kMaxSafeJsonInteger) {
      fail("JSON integer is outside the signed safe 53-bit range");
    }
    return JsonValue::integer_value(value);
  }
};

std::string escape_json_string(const std::string& value) {
  std::ostringstream output;
  output << '"';
  for (const unsigned char byte : value) {
    switch (byte) {
    case '"': output << "\\\""; break;
    case '\\': output << "\\\\"; break;
    case '\b': output << "\\b"; break;
    case '\f': output << "\\f"; break;
    case '\n': output << "\\n"; break;
    case '\r': output << "\\r"; break;
    case '\t': output << "\\t"; break;
    default:
      if (byte < 0x20U) {
        output << "\\u" << std::hex << std::setw(4) << std::setfill('0')
               << static_cast<int>(byte) << std::dec;
      } else {
        output << static_cast<char>(byte);
      }
    }
  }
  output << '"';
  return output.str();
}

std::string canonical_json(const JsonValue& value) {
  switch (value.type) {
  case JsonType::Null:
    return "null";
  case JsonType::Boolean:
    return value.boolean ? "true" : "false";
  case JsonType::Integer:
    return std::to_string(value.integer);
  case JsonType::String:
    return escape_json_string(value.string);
  case JsonType::Array: {
    std::string output = "[";
    for (std::size_t index = 0; index < value.array.size(); ++index) {
      if (index != 0) output.push_back(',');
      output += canonical_json(value.array[index]);
    }
    output.push_back(']');
    return output;
  }
  case JsonType::Object: {
    std::string output = "{";
    bool first = true;
    for (const auto& field : value.object) {
      if (!first) output.push_back(',');
      first = false;
      output += escape_json_string(field.first);
      output.push_back(':');
      output += canonical_json(field.second);
    }
    output.push_back('}');
    return output;
  }
  }
  throw std::runtime_error("unsupported canonical JSON type");
}

const std::map<std::string, JsonValue>& require_object(
    const JsonValue& value, const std::string& label) {
  if (value.type != JsonType::Object) {
    throw std::runtime_error(label + " must be one JSON object");
  }
  return value.object;
}

const std::vector<JsonValue>& require_array(
    const JsonValue& value, const std::string& label) {
  if (value.type != JsonType::Array) {
    throw std::runtime_error(label + " must be one JSON array");
  }
  return value.array;
}

const JsonValue& require_member(const std::map<std::string, JsonValue>& value,
                                const std::string& name,
                                const std::string& label) {
  const auto found = value.find(name);
  if (found == value.end()) {
    throw std::runtime_error(label + " is missing required field: " + name);
  }
  return found->second;
}

std::string require_string(const JsonValue& value, const std::string& label) {
  if (value.type != JsonType::String) {
    throw std::runtime_error(label + " must be one string");
  }
  return value.string;
}

std::int64_t require_integer(const JsonValue& value,
                             const std::string& label) {
  if (value.type != JsonType::Integer) {
    throw std::runtime_error(label + " must be one integer");
  }
  return value.integer;
}

bool require_boolean(const JsonValue& value, const std::string& label) {
  if (value.type != JsonType::Boolean) {
    throw std::runtime_error(label + " must be one boolean");
  }
  return value.boolean;
}

void require_string_equal(const std::map<std::string, JsonValue>& value,
                          const std::string& name,
                          const std::string& expected,
                          const std::string& label) {
  const std::string actual = require_string(require_member(value, name, label),
                                            label + "." + name);
  if (actual != expected) {
    throw std::runtime_error(label + "." + name + " is invalid");
  }
}

void require_members(const std::map<std::string, JsonValue>& value,
                     const std::vector<std::string>& fields,
                     const std::string& label) {
  for (const std::string& field : fields) {
    require_member(value, field, label);
  }
}

std::vector<std::string> object_names_from_array(
    const JsonValue& value, const std::string& label) {
  const auto& array = require_array(value, label);
  std::vector<std::string> output;
  output.reserve(array.size());
  for (std::size_t index = 0; index < array.size(); ++index) {
    const auto& object = require_object(
      array[index], label + "[" + std::to_string(index) + "]");
    output.push_back(require_string(
      require_member(object, "name", label), label + ".name"));
  }
  return output;
}

void validate_architecture(const std::map<std::string, JsonValue>& payload) {
  require_members(payload, {
    "abi", "required_capability_query_fields", "capabilities", "ownership",
    "asynchrony", "error_model", "selected_sp_semantics",
    "large_payload_policy", "compact_result_fields", "semantic_objects",
    "not_frozen"
  }, "architecture contract");
  const std::vector<std::string> expected = {
    "PreparedSGpuHandle", "TargetOptimizerStateHandle",
    "DeviceResidualHandle", "DcovComponentHandle", "LogicalCiBatchHandle",
    "CompactCiResult"
  };
  if (object_names_from_array(
        require_member(payload, "semantic_objects", "architecture contract"),
        "architecture semantic objects") != expected) {
    throw std::runtime_error("architecture semantic object order is invalid");
  }
  const auto& abi = require_object(
    require_member(payload, "abi", "architecture contract"),
    "architecture ABI");
  if (require_integer(require_member(abi, "major", "architecture ABI"),
                      "architecture ABI major") != 1) {
    throw std::runtime_error("architecture ABI major is invalid");
  }
  const auto& transfer = require_object(
    require_member(payload, "large_payload_policy", "architecture contract"),
    "large payload policy");
  require_string_equal(transfer, "residual_d2h", "forbidden",
                       "large payload policy");
  require_string_equal(transfer, "component_d2h", "forbidden",
                       "large payload policy");
}

void validate_numerical(const std::map<std::string, JsonValue>& payload) {
  require_members(payload, {
    "decimal_encoding", "precision", "comparison_metrics",
    "denominator_floors", "gam_formulas", "smoothing_parameter",
    "condition_buckets", "rank_policy", "dcov_formulas", "tolerances",
    "decision_semantics", "nonfinite_policy", "boundary_policy",
    "tolerance_change_policy"
  }, "numerical contract");
  const auto& precision = require_object(
    require_member(payload, "precision", "numerical contract"),
    "numerical precision");
  require_string_equal(precision, "storage", "IEEE-754-binary64",
                       "numerical precision");
  const auto& decision = require_object(
    require_member(payload, "decision_semantics", "numerical contract"),
    "decision semantics");
  require_string_equal(decision, "independent_when", "p_value >= alpha",
                       "decision semantics");
  if (require_integer(
        require_member(decision, "allowed_flip_count", "decision semantics"),
        "allowed flip count") != 0) {
    throw std::runtime_error("numerical decision flip allowance is invalid");
  }
  require_string_equal(payload, "nonfinite_policy",
                       "fail-closed-before-replay", "numerical contract");
  const std::vector<std::string> expected = {
    "CHOLESKY_BATCHED", "AUGMENTED_QR", "AUGMENTED_SVD", "AUGMENTED_SVD"
  };
  const auto& buckets = require_array(
    require_member(payload, "condition_buckets", "numerical contract"),
    "condition buckets");
  std::vector<std::string> solvers;
  for (const JsonValue& bucket_value : buckets) {
    const auto& bucket = require_object(bucket_value, "condition bucket");
    solvers.push_back(require_string(
      require_member(bucket, "solver", "condition bucket"),
      "condition bucket solver"));
  }
  if (solvers != expected) {
    throw std::runtime_error("numerical solver condition buckets are invalid");
  }
}

void validate_identity_contract(
    const std::map<std::string, JsonValue>& payload) {
  require_members(payload, {
    "canonicalization", "contract_snapshot", "producer_semantic_identity",
    "validator_attestation_identity", "volatile_execution_receipt",
    "artifact_semantic_identity", "publication"
  }, "artifact identity contract");
  const auto& canonical = require_object(
    require_member(payload, "canonicalization", "artifact identity contract"),
    "artifact canonicalization");
  require_string_equal(canonical, "schema", "full-cuda-ci-canonical-json-v1",
                       "artifact canonicalization");
  require_string_equal(canonical, "duplicate_object_keys", "forbidden",
                       "artifact canonicalization");
  require_string_equal(canonical, "json_numbers",
                       "signed-safe-53-bit-integers-only",
                       "artifact canonicalization");
}

void validate_reference_machine(
    const std::map<std::string, JsonValue>& payload) {
  require_members(payload, {"host", "cpu", "gpu", "software", "timing"},
                  "reference machine contract");
  const auto& gpu = require_object(
    require_member(payload, "gpu", "reference machine contract"),
    "reference GPU");
  require_string_equal(gpu, "model", "NVIDIA GeForce RTX 4090",
                       "reference GPU");
  if (require_integer(require_member(gpu, "device_id", "reference GPU"),
                      "reference GPU device") != 0) {
    throw std::runtime_error("reference GPU device is invalid");
  }
  require_string_equal(gpu, "compute_capability", "8.9", "reference GPU");
  const auto& timing = require_object(
    require_member(payload, "timing", "reference machine contract"),
    "reference timing");
  if (require_boolean(require_member(timing, "build_included", "timing"),
                      "build included")) {
    throw std::runtime_error("reference timing must exclude native build");
  }
}

void validate_performance_budget(
    const std::map<std::string, JsonValue>& payload) {
  require_members(payload, {
    "reference_machine_contract", "component_budgets", "feasibility",
    "promotion", "measurement"
  }, "performance budget contract");
  require_string_equal(payload, "reference_machine_contract",
                       "reference_machine_v1", "performance budget");
  const auto& budgets = require_object(
    require_member(payload, "component_budgets", "performance budget"),
    "component budgets");
  const std::vector<std::string> names = {
    "input_and_h2d", "native_setup", "gcv_selection",
    "fixed_sp_solve_and_residual", "dcov_component", "dcov_pair_and_gamma",
    "control_replay_and_packaging", "contingency"
  };
  std::int64_t total = 0;
  for (const std::string& name : names) {
    const auto& budget = require_object(
      require_member(budgets, name, "component budgets"), name);
    total += require_integer(
      require_member(budget, "warm_upper_bound_ms", name), name + " budget");
  }
  const auto& feasibility = require_object(
    require_member(payload, "feasibility", "performance budget"),
    "feasibility budget");
  const std::int64_t declared = require_integer(
    require_member(feasibility, "total_upper_bound_ms", "feasibility"),
    "feasibility total");
  if (total != 120000 || declared != total) {
    throw std::runtime_error("performance budget does not conserve 120 seconds");
  }
  const auto& dcov_component = require_object(
    require_member(budgets, "dcov_component", "component budgets"),
    "dcov component budget");
  const auto& dcov_pair = require_object(
    require_member(budgets, "dcov_pair_and_gamma", "component budgets"),
    "dcov pair budget");
  const std::int64_t dcov = require_integer(
    require_member(dcov_component, "warm_upper_bound_ms", "dcov component"),
    "dcov component budget") + require_integer(
      require_member(dcov_pair, "warm_upper_bound_ms", "dcov pair"),
      "dcov pair budget");
  if (dcov != require_integer(
        require_member(feasibility, "dcov_total_upper_bound_ms", "feasibility"),
        "dcov total budget")) {
    throw std::runtime_error("dCov performance budget is inconsistent");
  }
}

bool is_lower_sha256(const std::string& value) {
  if (value.size() != 64) return false;
  return std::all_of(value.begin(), value.end(), [](char byte) {
    return (byte >= '0' && byte <= '9') || (byte >= 'a' && byte <= 'f');
  });
}

void validate_development_corpus(
    const std::map<std::string, JsonValue>& payload) {
  require_members(payload, {
    "purpose", "source_identities", "canonical_counts",
    "required_risk_classes", "selection", "allowed_use",
    "promotion_holdout_claim"
  }, "development corpus contract");
  const auto& identities = require_object(
    require_member(payload, "source_identities", "development corpus"),
    "development source identities");
  for (const auto& field : identities) {
    if (!is_lower_sha256(require_string(field.second, field.first))) {
      throw std::runtime_error("development corpus SHA-256 is invalid");
    }
  }
  const auto& counts = require_object(
    require_member(payload, "canonical_counts", "development corpus"),
    "development corpus counts");
  if (require_integer(require_member(counts, "target_count", "counts"),
                      "target count") != 6143 ||
      require_integer(require_member(counts, "dcov_pair_count", "counts"),
                      "dCov pair count") != 3808 ||
      require_integer(
        require_member(counts, "near_alpha_pair_count", "counts"),
        "near-alpha pair count") != 1478) {
    throw std::runtime_error("development corpus counts are invalid");
  }
}

void validate_metamorphic(const std::map<std::string, JsonValue>& payload) {
  require_members(payload, {
    "purpose", "base_corpus", "global_invariants", "transformations",
    "minimum_coverage", "failure_policy"
  }, "metamorphic contract");
  require_string_equal(payload, "base_corpus",
                       "development_qualification_corpus_v1",
                       "metamorphic contract");
  const std::vector<std::string> expected = {
    "conditioning-column-permutation", "basis-sign-flip",
    "equivalent-orthogonal-rotation", "batch-split-merge",
    "stream-count-variation", "cache-capacity-variation",
    "standalone-versus-batched"
  };
  if (object_names_from_array(
        require_member(payload, "transformations", "metamorphic contract"),
        "metamorphic transformations") != expected) {
    throw std::runtime_error("metamorphic transformations are invalid");
  }
}

void validate_holdout(const std::map<std::string, JsonValue>& payload) {
  require_members(payload, {
    "holdout_id", "state", "ordinary_development_access",
    "payload_present_in_repository", "custody", "commitment",
    "release_protocol", "implementation_change_after_open_policy",
    "result_policy"
  }, "promotion holdout manifest");
  require_string_equal(payload, "state", "SEALED_NOT_RELEASED",
                       "promotion holdout");
  require_string_equal(payload, "ordinary_development_access", "forbidden",
                       "promotion holdout");
  if (require_boolean(
        require_member(payload, "payload_present_in_repository", "holdout"),
        "holdout payload presence")) {
    throw std::runtime_error("promotion holdout payload must not be in repository");
  }
  const auto& custody = require_object(
    require_member(payload, "custody", "promotion holdout"),
    "promotion holdout custody");
  const auto& commitment = require_object(
    require_member(payload, "commitment", "promotion holdout"),
    "promotion holdout commitment");
  require_string_equal(
    commitment, "identity_formula",
    "sha256(canonical(custody_authority,holdout_id,payload_present_in_repository,release_phase,state))",
    "promotion holdout commitment");
  std::map<std::string, JsonValue> commitment_fields;
  commitment_fields.emplace(
    "custody_authority",
    JsonValue::string_value(require_string(
      require_member(custody, "authority", "promotion holdout custody"),
      "promotion holdout custody authority")));
  commitment_fields.emplace(
    "holdout_id",
    JsonValue::string_value(require_string(
      require_member(payload, "holdout_id", "promotion holdout"),
      "promotion holdout ID")));
  commitment_fields.emplace(
    "payload_present_in_repository",
    JsonValue::boolean_value(require_boolean(
      require_member(payload, "payload_present_in_repository", "holdout"),
      "holdout payload presence")));
  commitment_fields.emplace(
    "release_phase",
    JsonValue::string_value(require_string(
      require_member(custody, "release_phase", "promotion holdout custody"),
      "promotion holdout release phase")));
  commitment_fields.emplace(
    "state",
    JsonValue::string_value(require_string(
      require_member(payload, "state", "promotion holdout"),
      "promotion holdout state")));
  const std::string expected_commitment = full_cuda_ci_sha256_utf8(
    canonical_json(JsonValue::object_value(std::move(commitment_fields))));
  if (require_string(
        require_member(commitment, "manifest_identity_sha256",
                       "promotion holdout commitment"),
        "promotion holdout commitment SHA-256") != expected_commitment) {
    throw std::runtime_error("promotion holdout commitment hash mismatch");
  }
}

void validate_known_payload(const std::string& name,
                            const std::map<std::string, JsonValue>& payload) {
  if (name == "architecture_contract_v1") {
    validate_architecture(payload);
  } else if (name == "numerical_contract_v1") {
    validate_numerical(payload);
  } else if (name == "artifact_identity_contract_v1") {
    validate_identity_contract(payload);
  } else if (name == "reference_machine_v1") {
    validate_reference_machine(payload);
  } else if (name == "performance_budget_v1") {
    validate_performance_budget(payload);
  } else if (name == "development_qualification_corpus_v1") {
    validate_development_corpus(payload);
  } else if (name == "metamorphic_contract_v1") {
    validate_metamorphic(payload);
  } else if (name == "promotion_holdout_manifest_v1") {
    validate_holdout(payload);
  }
}

constexpr std::array<std::uint32_t, 64> kSha256Constants = {{
  0x428a2f98U, 0x71374491U, 0xb5c0fbcfU, 0xe9b5dba5U,
  0x3956c25bU, 0x59f111f1U, 0x923f82a4U, 0xab1c5ed5U,
  0xd807aa98U, 0x12835b01U, 0x243185beU, 0x550c7dc3U,
  0x72be5d74U, 0x80deb1feU, 0x9bdc06a7U, 0xc19bf174U,
  0xe49b69c1U, 0xefbe4786U, 0x0fc19dc6U, 0x240ca1ccU,
  0x2de92c6fU, 0x4a7484aaU, 0x5cb0a9dcU, 0x76f988daU,
  0x983e5152U, 0xa831c66dU, 0xb00327c8U, 0xbf597fc7U,
  0xc6e00bf3U, 0xd5a79147U, 0x06ca6351U, 0x14292967U,
  0x27b70a85U, 0x2e1b2138U, 0x4d2c6dfcU, 0x53380d13U,
  0x650a7354U, 0x766a0abbU, 0x81c2c92eU, 0x92722c85U,
  0xa2bfe8a1U, 0xa81a664bU, 0xc24b8b70U, 0xc76c51a3U,
  0xd192e819U, 0xd6990624U, 0xf40e3585U, 0x106aa070U,
  0x19a4c116U, 0x1e376c08U, 0x2748774cU, 0x34b0bcb5U,
  0x391c0cb3U, 0x4ed8aa4aU, 0x5b9cca4fU, 0x682e6ff3U,
  0x748f82eeU, 0x78a5636fU, 0x84c87814U, 0x8cc70208U,
  0x90befffaU, 0xa4506cebU, 0xbef9a3f7U, 0xc67178f2U
}};

std::uint32_t rotate_right(std::uint32_t value, unsigned int bits) {
  return (value >> bits) | (value << (32U - bits));
}

std::array<std::uint8_t, 32> sha256_bytes(const std::string& input) {
  std::vector<std::uint8_t> message(input.begin(), input.end());
  const std::uint64_t bit_length =
    static_cast<std::uint64_t>(message.size()) * 8U;
  message.push_back(0x80U);
  while ((message.size() % 64U) != 56U) message.push_back(0U);
  for (int shift = 56; shift >= 0; shift -= 8) {
    message.push_back(static_cast<std::uint8_t>(bit_length >> shift));
  }

  std::array<std::uint32_t, 8> hash = {{
    0x6a09e667U, 0xbb67ae85U, 0x3c6ef372U, 0xa54ff53aU,
    0x510e527fU, 0x9b05688cU, 0x1f83d9abU, 0x5be0cd19U
  }};
  for (std::size_t offset = 0; offset < message.size(); offset += 64U) {
    std::array<std::uint32_t, 64> words{};
    for (int index = 0; index < 16; ++index) {
      const std::size_t base = offset + static_cast<std::size_t>(index) * 4U;
      words[index] =
        (static_cast<std::uint32_t>(message[base]) << 24U) |
        (static_cast<std::uint32_t>(message[base + 1U]) << 16U) |
        (static_cast<std::uint32_t>(message[base + 2U]) << 8U) |
        static_cast<std::uint32_t>(message[base + 3U]);
    }
    for (int index = 16; index < 64; ++index) {
      const std::uint32_t s0 = rotate_right(words[index - 15], 7U) ^
        rotate_right(words[index - 15], 18U) ^ (words[index - 15] >> 3U);
      const std::uint32_t s1 = rotate_right(words[index - 2], 17U) ^
        rotate_right(words[index - 2], 19U) ^ (words[index - 2] >> 10U);
      words[index] = words[index - 16] + s0 + words[index - 7] + s1;
    }
    std::uint32_t a = hash[0];
    std::uint32_t b = hash[1];
    std::uint32_t c = hash[2];
    std::uint32_t d = hash[3];
    std::uint32_t e = hash[4];
    std::uint32_t f = hash[5];
    std::uint32_t g = hash[6];
    std::uint32_t h = hash[7];
    for (int index = 0; index < 64; ++index) {
      const std::uint32_t sum1 = rotate_right(e, 6U) ^
        rotate_right(e, 11U) ^ rotate_right(e, 25U);
      const std::uint32_t choice = (e & f) ^ ((~e) & g);
      const std::uint32_t temporary1 =
        h + sum1 + choice + kSha256Constants[index] + words[index];
      const std::uint32_t sum0 = rotate_right(a, 2U) ^
        rotate_right(a, 13U) ^ rotate_right(a, 22U);
      const std::uint32_t majority = (a & b) ^ (a & c) ^ (b & c);
      const std::uint32_t temporary2 = sum0 + majority;
      h = g;
      g = f;
      f = e;
      e = d + temporary1;
      d = c;
      c = b;
      b = a;
      a = temporary1 + temporary2;
    }
    hash[0] += a;
    hash[1] += b;
    hash[2] += c;
    hash[3] += d;
    hash[4] += e;
    hash[5] += f;
    hash[6] += g;
    hash[7] += h;
  }
  std::array<std::uint8_t, 32> output{};
  for (std::size_t index = 0; index < hash.size(); ++index) {
    output[index * 4U] = static_cast<std::uint8_t>(hash[index] >> 24U);
    output[index * 4U + 1U] = static_cast<std::uint8_t>(hash[index] >> 16U);
    output[index * 4U + 2U] = static_cast<std::uint8_t>(hash[index] >> 8U);
    output[index * 4U + 3U] = static_cast<std::uint8_t>(hash[index]);
  }
  return output;
}

}  // namespace

std::string full_cuda_ci_sha256_utf8(const std::string& value) {
  const auto bytes = sha256_bytes(value);
  std::ostringstream output;
  output << std::hex << std::setfill('0');
  for (const std::uint8_t byte : bytes) {
    output << std::setw(2) << static_cast<int>(byte);
  }
  return output.str();
}

FullCudaCiContractIdentity full_cuda_ci_contract_identity(
    const std::string& json,
    const std::string& expected_contract_name) {
  if (expected_contract_name.empty()) {
    throw std::runtime_error("expected contract name must not be empty");
  }
  JsonValue document = JsonParser(json).parse();
  const auto& root = require_object(document, "tracked contract envelope");
  const auto name_field = root.find("contract_name");
  if (name_field == root.end() || name_field->second.type != JsonType::String ||
      name_field->second.string != expected_contract_name) {
    throw std::runtime_error("tracked contract name mismatch");
  }
  const std::vector<std::string> envelope_fields = {
    "campaign", "contract_name", "contract_schema_version",
    "phase_introduced", "semantic_version", "payload"
  };
  require_members(root, envelope_fields, "tracked contract envelope");
  if (root.size() != envelope_fields.size()) {
    throw std::runtime_error("tracked contract envelope has unknown fields");
  }
  require_string_equal(root, "campaign", "full-cuda-legacy-compatible-ci",
                       "tracked contract envelope");
  require_string_equal(root, "contract_schema_version",
                       "full-cuda-ci-tracked-contract-v1",
                       "tracked contract envelope");
  require_string_equal(root, "phase_introduced", "3.5",
                       "tracked contract envelope");
  const auto& version = require_object(
    require_member(root, "semantic_version", "tracked contract envelope"),
    "tracked contract semantic version");
  if (version.size() != 3 || version.find("major") == version.end() ||
      version.find("minor") == version.end() ||
      version.find("patch") == version.end()) {
    throw std::runtime_error("tracked contract semantic version is malformed");
  }
  const std::int64_t major = require_integer(
    require_member(version, "major", "semantic version"), "semantic major");
  const std::int64_t minor = require_integer(
    require_member(version, "minor", "semantic version"), "semantic minor");
  const std::int64_t patch = require_integer(
    require_member(version, "patch", "semantic version"), "semantic patch");
  if (major < 1 || minor < 0 || patch < 0 ||
      major > std::numeric_limits<int>::max() ||
      minor > std::numeric_limits<int>::max() ||
      patch > std::numeric_limits<int>::max()) {
    throw std::runtime_error("tracked contract semantic version is invalid");
  }
  const auto& payload = require_object(
    require_member(root, "payload", "tracked contract envelope"),
    "tracked contract payload");
  validate_known_payload(expected_contract_name, payload);

  FullCudaCiContractIdentity identity;
  identity.contract_name = expected_contract_name;
  identity.contract_schema_version = "full-cuda-ci-tracked-contract-v1";
  identity.semantic_major = static_cast<int>(major);
  identity.semantic_minor = static_cast<int>(minor);
  identity.semantic_patch = static_cast<int>(patch);
  identity.canonical_json = canonical_json(document);
  identity.sha256 = full_cuda_ci_sha256_utf8(identity.canonical_json);
  return identity;
}

}  // namespace fastkpc
