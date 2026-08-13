#pragma once

// The two strings every front end has to produce: a file size and a duration.
//
// Both used to exist three times over -- once in the command line tool, once
// in the pipeline's log lines, once in the window -- with the thresholds and
// the rounding drifting between them. A user comparing what the log said with
// what the queue said should not be reading two different opinions about the
// same number.

#include <cstdint>
#include <cstdio>
#include <string>

namespace gsic {

// "812.4 KB" / "1.90 MB". Kilobytes below a megabyte, because a compressed
// image is usually in that range and "0.79 MB" reads worse than "812.4 KB".
inline std::string human_size(std::int64_t bytes) {
    char buf[32];
    if (bytes >= 1024 * 1024)
        std::snprintf(buf, sizeof(buf), "%.2f MB", double(bytes) / (1024.0 * 1024.0));
    else
        std::snprintf(buf, sizeof(buf), "%.1f KB", double(bytes) / 1024.0);
    return buf;
}

// "12s" / "1m 40s". Rounded coarsely on purpose: a projection accurate to the
// second would be claiming a precision it does not have.
inline std::string human_duration(double seconds) {
    char buf[32];
    if (!(seconds > 0.0)) return "<1s";   // also catches NaN
    if (seconds < 1.0) return "<1s";
    if (seconds < 60.0) {
        std::snprintf(buf, sizeof(buf), "%ds", int(seconds + 0.5));
        return buf;
    }
    if (seconds >= 100.0 * 3600.0) return "a very long time";
    const int total = int(seconds + 0.5);
    std::snprintf(buf, sizeof(buf), "%dm %02ds", total / 60, total % 60);
    return buf;
}

} // namespace gsic
