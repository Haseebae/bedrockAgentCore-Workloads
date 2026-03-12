"""
Log Workloads Benchmark Queries
"""

# ---------------------------------------------------------------------------
# Workload 1 – Apache / mod_jk error log (example.log)
# ---------------------------------------------------------------------------
S3_LOG_PATH1 = "s3://mcp-log-analytics/0b5d5771-16aa-4e80-89d7-cbb1672c8cda/example.log"

_w1_q1 = (
    f"PASS THE 'filename' PARAMETER TO THE TOOLS AS 'example.log'. "
    f"In '{S3_LOG_PATH1}', the FILE NAME TO PASS TO THE TOOLS IS 'example.log'. "
    f"DO NOT FORGET TO INPUT THE ENTIRE LOG FILE NAME TO THE TOOLS IN S3 URI FORMAT, the filename should end with .log. "
    f"How many times does the error 'mod_jk child workerEnv in error state 6' and "
    f"'mod_jk child workerEnv in error state 7' occur in the log file? "
    f"\n\nTell this count for 'mod_jk child workerEnv in error state 8' and "
    f"'mod_jk child workerEnv in error state 9' as well. "
    f"DO NOT HALLUCINATE ANYTHING, ONLY USE THE TOOLS PROVIDED TO YOU. "
    f"ONLY RETURN S3 PATHS IF YOU ACTUALLY SAVE IT, NOTHING OTHERWISE. "
    f"PASS ALL THE Required Input parameters correctly to the tools."
)

_w1_q2 = (
    f"PASS THE 'filename' PARAMETER TO THE TOOLS AS 'example.log'. "
    f"For the error that occurs most frequently ('mod_jk child workerEnv in error state 6'), "
    f"give me the mean and standard deviation of line numbers where I can find this error in the log file. "
    f"PASS ALL THE Required Input parameters correctly to the tools. "
    f"PASS THE FILENAME TO THE TOOLS AS 'example.log'."
)

_w1_q3 = (
    f"PASS THE 'filename' PARAMETER TO THE TOOLS AS 'example.log'. "
    f"For the error that occurs most frequently ('mod_jk child workerEnv in error state 6'):\n"
    f"1. Find the minimum line number where this error occurs\n"
    f"2. Find the maximum line number where this error occurs\n"
    f"3. Find the mean of line numbers where this error occurs\n"
    f"4. Find the median of line numbers where this error occurs\n"
    f"5. Create a bar chart comparing the minimum, maximum, mean, and median line numbers\n"
    f"6. Store STRICTLY in S3, NOT IN LOCAL FILE SYSTEM AT ALL. "
    f"Return the CORRECT OUTPUT FILE that the tool call tells you, NOT any random name you thought of.\n"
    f"7. Compare the mean values for line numbers for 'mod_jk child workerEnv in error state 6' "
    f"and 'mod_jk child workerEnv in error state 9'. "
    f"PASS ALL THE Required Input parameters correctly to the tools. "
    f"PASS THE FILENAME TO THE TOOLS AS 'example.log'."
)

# ---------------------------------------------------------------------------
# Workload 2 – Hadoop / YARN NodeManager log (example2.log)
# ---------------------------------------------------------------------------
S3_LOG_PATH2 = "s3://mcp-log-analytics/0b5d5771-16aa-4e80-89d7-cbb1672c8cda/example2.log"

_w2_q1 = (
    f"Count how many times each of these errors occurs in the log file at '{S3_LOG_PATH2}':\n"
    f"1. 'Failed to renew lease'\n"
    f"2. 'ERROR IN CONTACTING RM'\n"
    f"3. 'Retrying connect to server'\n"
    f"4. 'Address change detected'\n\n"
    f"For each error pattern:\n"
    f"- Find all line numbers where the error occurs in the log file\n"
    f"- Count the total number of occurrences\n"
    f"Report all four counts."
)

_w2_q2 = (
    f"PASS ALL THE Required Input parameters correctly to the tools. "
    f"For the error that occurs most frequently ('Failed to renew lease'), "
    f"give me the mean and standard deviation of line numbers where I can find this error in the log file. "
    f"Use proper tools for length calculation; for all counting or math use tools instead of doing it yourself. "
    f"NO COUNT IS ZERO IF YOU ARE THINKING IT IS — RE-THINK AND REVISIT YOUR ANSWERS AND GIVE CORRECT ANSWERS."
)

_w2_q3 = (
    f"For the error that occurs most frequently ('Failed to renew lease'):\n"
    f"1. Find the minimum line number where this error occurs\n"
    f"2. Find the maximum line number where this error occurs\n"
    f"3. Find the mean of line numbers where this error occurs\n"
    f"4. Find the median of line numbers where this error occurs\n"
    f"5. Create a bar chart comparing the minimum, maximum, mean, and median line numbers\n"
    f"6. Store STRICTLY in S3, NOT IN LOCAL FILE SYSTEM AT ALL. "
    f"Return the CORRECT OUTPUT FILE that the tool call tells you, NOT any random name you thought of.\n"
    f"7. Compare the mean values for line numbers for 'Failed to renew lease' and 'Address change detected'. "
    f"Use proper tools for length calculation; for all counting or math use tools instead of doing it yourself. "
    f"NO COUNT IS ZERO IF YOU ARE THINKING IT IS — RE-THINK AND REVISIT YOUR ANSWERS AND GIVE CORRECT ANSWERS."
)

# ---------------------------------------------------------------------------
# Workload 3 – SSH / auth log (example3.log)
# ---------------------------------------------------------------------------
S3_LOG_PATH3 = "s3://mcp-log-analytics/0b5d5771-16aa-4e80-89d7-cbb1672c8cda/example3.log"

_w3_q1 = (
    f"Count how many times each of these errors occurs in the log file at '{S3_LOG_PATH3}':\n"
    f"1. 'authentication failure'\n"
    f"2. 'Received disconnect'\n"
    f"3. 'Failed password for root'\n"
    f"4. 'Failed password for invalid user'\n\n"
    f"For each error pattern:\n"
    f"- Find all line numbers where the error occurs in the log file\n"
    f"- Count the total number of occurrences\n"
    f"Report all four counts."
)

_w3_q2 = (
    f"For the error 'authentication failure', "
    f"calculate and report the mean and standard deviation of line numbers where this error occurs."
)

_w3_q3 = (
    f"For the error 'authentication failure':\n"
    f"Calculate the minimum, maximum, mean, and median of line numbers where this error occurs. "
    f"Create a bar chart comparing these four values and save it to S3. "
    f"Also calculate and compare the mean line numbers for 'authentication failure' "
    f"and 'Failed password for invalid user'."
)

# TOTAL LOG QUERIES
LOG_QUERIES = [
    [
        {"name": "Turn 1: Count error states",              "query": _w1_q1},
        {"name": "Turn 2: Mean & std-dev of line numbers",  "query": _w1_q2},
        {"name": "Turn 3: Min/max/mean/median + chart",     "query": _w1_q3},
    ],
    [
        {"name": "Turn 1: Count error states",              "query": _w2_q1},
        {"name": "Turn 2: Mean & std-dev of line numbers",  "query": _w2_q2},
        {"name": "Turn 3: Min/max/mean/median + chart",     "query": _w2_q3},
    ],
    [
        {"name": "Turn 1: Count error states",              "query": _w3_q1},
        {"name": "Turn 2: Mean & std-dev of line numbers",  "query": _w3_q2},
        {"name": "Turn 3: Min/max/mean/median + chart",     "query": _w3_q3},
    ],
]

def get_log_workload():
    return LOG_QUERIES