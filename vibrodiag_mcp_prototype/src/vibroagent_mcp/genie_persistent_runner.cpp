// Persistent Qualcomm Genie dialog runner used by genie_openai_server.py.
//
// Protocol:
//   child writes protocol responses to fd 3, keeping stdout/stderr free for SDK logs.
//   parent sends "QUERY <n>\n<prompt-bytes>\n" or "QUIT\n" on stdin.
//   child replies "OK <n> <prompt_tokens> <completion_tokens> <total_us> <generation_us> <ttft_us>\n<answer-bytes>\n".
//   child replies "ERR <n>\n<message-bytes>\n" on failure.

#include <Genie/GenieDialog.h>
#include <Genie/GenieTokenizer.h>

#include <cerrno>
#include <chrono>
#include <cctype>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <unistd.h>

namespace {

using Clock = std::chrono::steady_clock;

thread_local std::vector<char> g_alloc_buffer;

void tokenizer_alloc_callback(const size_t size, const char** allocated_data) {
  g_alloc_buffer.resize(size);
  *allocated_data = g_alloc_buffer.data();
}

bool write_all(int fd, const char* data, size_t size) {
  while (size > 0) {
    ssize_t written = ::write(fd, data, size);
    if (written < 0) {
      if (errno == EINTR) {
        continue;
      }
      return false;
    }
    if (written == 0) {
      return false;
    }
    data += written;
    size -= static_cast<size_t>(written);
  }
  return true;
}

bool write_string(int fd, const std::string& value) {
  return write_all(fd, value.data(), value.size());
}

bool send_error(int fd, const std::string& message) {
  std::ostringstream header;
  header << "ERR " << message.size() << "\n";
  return write_string(fd, header.str()) && write_string(fd, message) && write_string(fd, "\n");
}

std::string read_file(const std::string& path) {
  std::ifstream input(path, std::ios::binary);
  if (!input) {
    throw std::runtime_error("Could not open Genie config: " + path);
  }
  std::ostringstream buffer;
  buffer << input.rdbuf();
  return buffer.str();
}

std::string status_message(const std::string& operation, Genie_Status_t status) {
  std::ostringstream message;
  message << operation << " failed with Genie status " << status;
  return message.str();
}

void replace_all(std::string& text, const std::string& needle, const std::string& replacement) {
  if (needle.empty()) {
    return;
  }
  size_t pos = 0;
  while ((pos = text.find(needle, pos)) != std::string::npos) {
    text.replace(pos, needle.size(), replacement);
    pos += replacement.size();
  }
}

std::string trim_and_strip_stops(std::string text) {
  replace_all(text, "<|im_end|>", "");
  replace_all(text, "<|endoftext|>", "");

  size_t start = 0;
  while (start < text.size() && std::isspace(static_cast<unsigned char>(text[start]))) {
    ++start;
  }
  size_t end = text.size();
  while (end > start && std::isspace(static_cast<unsigned char>(text[end - 1]))) {
    --end;
  }
  return text.substr(start, end - start);
}

struct QueryState {
  std::string output;
  Clock::time_point started;
  Clock::time_point first_output;
  Clock::time_point finished;
  bool saw_output = false;
};

void query_callback(const char* response,
                    const GenieDialog_SentenceCode_t sentence_code,
                    const void* user_data) {
  auto* state = static_cast<QueryState*>(const_cast<void*>(user_data));
  if (!state) {
    return;
  }
  if (response && response[0] != '\0') {
    if (!state->saw_output) {
      state->first_output = Clock::now();
      state->saw_output = true;
    }
    state->output += response;
  }
  if (sentence_code == GENIE_DIALOG_SENTENCE_COMPLETE ||
      sentence_code == GENIE_DIALOG_SENTENCE_END ||
      sentence_code == GENIE_DIALOG_SENTENCE_ABORT) {
    state->finished = Clock::now();
  }
}

int64_t elapsed_us(Clock::time_point start, Clock::time_point end) {
  return std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
}

struct QueryResult {
  std::string text;
  int prompt_tokens = -1;
  int completion_tokens = -1;
  int64_t total_us = 0;
  int64_t generation_us = 0;
  int64_t time_to_first_token_us = -1;
};

class PersistentDialog {
 public:
  explicit PersistentDialog(const std::string& config_path) {
    const std::string config_json = read_file(config_path);
    Genie_Status_t status = GenieDialogConfig_createFromJson(config_json.c_str(), &config_);
    if (status != GENIE_STATUS_SUCCESS || !config_) {
      throw std::runtime_error(status_message("GenieDialogConfig_createFromJson", status));
    }

    status = GenieDialog_create(config_, &dialog_);
    if (status != GENIE_STATUS_SUCCESS || !dialog_) {
      throw std::runtime_error(status_message("GenieDialog_create", status));
    }

    status = GenieDialog_getTokenizer(dialog_, &tokenizer_);
    if (status != GENIE_STATUS_SUCCESS) {
      tokenizer_ = nullptr;
    }
  }

  ~PersistentDialog() {
    if (dialog_) {
      GenieDialog_free(dialog_);
    }
    if (config_) {
      GenieDialogConfig_free(config_);
    }
  }

  PersistentDialog(const PersistentDialog&) = delete;
  PersistentDialog& operator=(const PersistentDialog&) = delete;

  QueryResult query(const std::string& prompt) {
    Genie_Status_t status = GenieDialog_reset(dialog_);
    if (status != GENIE_STATUS_SUCCESS) {
      throw std::runtime_error(status_message("GenieDialog_reset", status));
    }

    QueryState state;
    state.started = Clock::now();
    status = GenieDialog_query(dialog_,
                               prompt.empty() ? nullptr : prompt.c_str(),
                               GENIE_DIALOG_SENTENCE_COMPLETE,
                               query_callback,
                               &state);
    state.finished = Clock::now();
    if (status != GENIE_STATUS_SUCCESS && status != GENIE_STATUS_WARNING_CONTEXT_EXCEEDED) {
      throw std::runtime_error(status_message("GenieDialog_query", status));
    }

    QueryResult result;
    result.text = trim_and_strip_stops(state.output);
    result.prompt_tokens = count_tokens(prompt);
    result.completion_tokens = count_tokens(result.text);
    result.total_us = std::max<int64_t>(0, elapsed_us(state.started, state.finished));
    if (state.saw_output) {
      result.time_to_first_token_us = std::max<int64_t>(0, elapsed_us(state.started, state.first_output));
      result.generation_us = std::max<int64_t>(0, elapsed_us(state.first_output, state.finished));
    }
    if (result.generation_us <= 0) {
      result.generation_us = result.total_us;
    }
    return result;
  }

 private:
  int count_tokens(const std::string& text) {
    if (!tokenizer_ || text.empty()) {
      return text.empty() ? 0 : -1;
    }
    const int32_t* token_ids = nullptr;
    uint32_t num_token_ids = 0;
    g_alloc_buffer.clear();
    const Genie_Status_t status =
        GenieTokenizer_encode(tokenizer_, text.c_str(), tokenizer_alloc_callback, &token_ids, &num_token_ids);
    if (status != GENIE_STATUS_SUCCESS) {
      return -1;
    }
    return static_cast<int>(num_token_ids);
  }

  GenieDialogConfig_Handle_t config_ = nullptr;
  GenieDialog_Handle_t dialog_ = nullptr;
  GenieTokenizer_Handle_t tokenizer_ = nullptr;
};

bool send_ok(int fd, const QueryResult& result) {
  std::ostringstream header;
  header << "OK " << result.text.size() << " " << result.prompt_tokens << " "
         << result.completion_tokens << " " << result.total_us << " " << result.generation_us << " "
         << result.time_to_first_token_us << "\n";
  return write_string(fd, header.str()) && write_string(fd, result.text) && write_string(fd, "\n");
}

bool handle_query_line(const std::string& line, PersistentDialog& dialog, int protocol_fd) {
  if (line == "QUIT") {
    return false;
  }
  if (line.rfind("QUERY ", 0) != 0) {
    send_error(protocol_fd, "unknown command: " + line);
    return true;
  }

  const std::string size_text = line.substr(6);
  size_t prompt_size = 0;
  try {
    prompt_size = static_cast<size_t>(std::stoull(size_text));
  } catch (const std::exception&) {
    send_error(protocol_fd, "invalid query size: " + size_text);
    return true;
  }

  std::string prompt(prompt_size, '\0');
  if (prompt_size > 0) {
    std::cin.read(&prompt[0], static_cast<std::streamsize>(prompt_size));
    if (static_cast<size_t>(std::cin.gcount()) != prompt_size) {
      send_error(protocol_fd, "failed to read complete prompt payload");
      return false;
    }
  }
  const int delimiter = std::cin.get();
  if (delimiter != '\n' && delimiter != EOF) {
    send_error(protocol_fd, "invalid query delimiter");
    return true;
  }

  try {
    const QueryResult result = dialog.query(prompt);
    if (!send_ok(protocol_fd, result)) {
      return false;
    }
  } catch (const std::exception& exc) {
    send_error(protocol_fd, exc.what());
  }
  return true;
}

}  // namespace

int main(int argc, char** argv) {
  int protocol_fd = 3;
  if (const char* raw_fd = std::getenv("GENIE_PERSISTENT_PROTOCOL_FD")) {
    try {
      protocol_fd = std::stoi(raw_fd);
    } catch (const std::exception&) {
      protocol_fd = 3;
    }
  }
  if (argc != 2) {
    send_error(protocol_fd, "usage: genie_persistent_runner <genie_config.json>");
    return 2;
  }

  try {
    PersistentDialog dialog(argv[1]);
    if (!write_string(protocol_fd, "READY\n")) {
      return 1;
    }

    std::string line;
    while (std::getline(std::cin, line)) {
      if (!handle_query_line(line, dialog, protocol_fd)) {
        break;
      }
    }
  } catch (const std::exception& exc) {
    send_error(protocol_fd, exc.what());
    return 1;
  }
  return 0;
}
