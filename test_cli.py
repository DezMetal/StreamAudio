import os
from main import Voice

# Create a dummy test file
with open("test_file.txt", "w") as f:
    f.write("This is a test file.")

V = Voice(headless=True)

# Test case 1: File not found
# Verify that audio is still generated for the fallback text.
audio_bytes_not_found = V.say('[file="nonexistent.txt"] Fallback text. [bytes]', speak_override=False)
assert audio_bytes_not_found is not None
assert len(audio_bytes_not_found) > 0

# Test case 2: File found
# Verify that including a file results in a different (longer) audio output.
audio_bytes_found = V.say('[file="test_file.txt"] Fallback text. [bytes]', speak_override=False)
assert audio_bytes_found is not None
assert len(audio_bytes_found) > len(audio_bytes_not_found)

print("All tests passed!")

# Clean up the dummy file
os.remove("test_file.txt")