import os
from main import Voice

# Create dummy test files
with open("test_file_A.txt", "w") as f:
    f.write("This is the content of file A.")
with open("test_file_B.txt", "w") as f:
    f.write("This is the content of file B.")

V = Voice(headless=True)

# Test Case 1: File Not Found
# Synthesize text where the file tag is invalid.
# The program should recover and synthesize the surrounding text.
V.reset_dynamics()
audio_bytes_not_found = V.say('Before. [file="nonexistent.txt"] After. [bytes]', speak_override=False)
assert audio_bytes_not_found is not None and len(audio_bytes_not_found) > 0

# Test Case 2: Single File Found
# Synthesize text with a valid file tag.
# The resulting audio should be longer than the audio from the not-found case.
V.reset_dynamics()
audio_bytes_single_found = V.say('Before. [file="test_file_A.txt"] After. [bytes]', speak_override=False)
assert audio_bytes_single_found is not None
assert len(audio_bytes_single_found) > len(audio_bytes_not_found)

# Test Case 3: Multiple Files Found
# Synthesize text with two valid file tags.
# The resulting audio should be longer than the audio with just one file.
V.reset_dynamics()
audio_bytes_multiple_found = V.say('File A is [file="test_file_A.txt"] and File B is [file="test_file_B.txt"]. [bytes]', speak_override=False)
assert audio_bytes_multiple_found is not None
assert len(audio_bytes_multiple_found) > len(audio_bytes_single_found)

print("All tests passed!")

# Clean up the dummy files
os.remove("test_file_A.txt")
os.remove("test_file_B.txt")