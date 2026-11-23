# Log

Newest last. Anyone may append: the operator, the Dash, an agent.

## 2026-11-22 18:42 · claude

Registered as a project.

## 2026-11-22 18:42 · claude

status: — → the lab's speech origin: Piper TTS with dynamics and RVC voice conversion; vendored into Aether and DNetAva
Carded so jobs that belong in the origin (not a vendored copy) have a place to land; seeded by claude, confirm or correct.

## 2026-11-22 · claude

Speech latency fixed in `main.py` `Voice.say_fast` -- the path Aether speaks
on. `SpeechQueue` has the same chunking and had the Aug 14 first-chunk work,
but nothing in the lab plays through it (DLLama uses it for buffers only), so
that fix was invisible to every listener.

Two causes, both about WHEN the work happened:

1. Conversion and playback shared the calling thread. `st.write()` blocks
   until the device takes the audio, so piper and RVC for chunk N+1 did not
   start until chunk N had drained. Each join cost a conversion in silence,
   and piper yields a chunk per sentence -- so it was heard at punctuation.
   Playback now runs on its own thread fed by an unbounded queue; a bound
   would put the producer back behind the speakers.
2. A flat 1.5s of speech was accumulated before the first conversion pass.
   The leading chunk is now 0.35s and grows back to 1.5s once audio is
   flowing, matching what SpeechQueue already did.

Also: the RVC-with-overlap-context pass existed three times (twice inside
say_fast, once in SpeechQueue). One `_rvc_pass` now, and the chunk-size
numbers are module level.

Not committed. Landing it in Aether and DNetAva needs a push here, then a
pull and a submodule pointer bump in each consumer.

## 2026-11-22 · claude

The output device is no longer opened and closed per call. `say_fast` paid a
device setup at the front of every utterance, so a reply arriving as several
fragments paid it per fragment, exactly where it is heard. There is now one
stream on `Voice`, held while she is talking and released 20s after she stops
by a timer the next utterance cancels -- not a loop watching for silence,
because the daemon stays hot for hours and must not hold the device through
them. `_open_stream` is gone; `say`, its RVC playback and `say_fast` all take
the same stream, and its dead `check_interrupted_callback` went with it.

Consequence handled: `stop()` now aborts the stream. With the device held open
a drain would have played out the rest of the sentence she was told to
abandon, and the next utterance would have queued behind it.

Measured while reading piper: `PiperVoice.synthesize` yields ONE CHUNK PER
SENTENCE. The accumulate target is only tested after a whole chunk arrives, so
no value of it can make the leading audio shorter than the first sentence --
the "short first chunk" idea cannot do what its comment claimed, here or in
SpeechQueue. What the small leading target really does is stop a short opening
sentence from waiting for the second and third to be synthesized before
anything is converted. The comments now say that instead.
