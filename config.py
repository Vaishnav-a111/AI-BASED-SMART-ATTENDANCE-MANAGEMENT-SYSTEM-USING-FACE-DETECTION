# Face recognition configuration
# Adjust these values to tune matching sensitivity and performance.

# Distance tolerance for a match (lower = stricter).
# 0.55 with the large model + average-distance matching handles:
#  - glasses / no-glasses variation
#  - different facial expressions and head angles
#  - webcam lighting differences
# Lowering below 0.50 causes too many "Face undetected" for real students.
FACE_MATCH_TOLERANCE = 0.55

# Margin: the gap required between the best and second-best student.
# 0.08 rejects genuinely ambiguous matches while not being overly strict.
FACE_MATCH_MARGIN = 0.08

# Number of jitter iterations for live frame encoding.
# 3 gives better quality than 2 without a big speed penalty.
LIVE_ENCODING_JITTERS = 3

# Optional: maximum acceptable latency per frame in milliseconds (for monitoring).
MAX_LATENCY_MS = 200
