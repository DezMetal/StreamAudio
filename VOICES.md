# Voice models

**StreamAudio does not ship with voice models, and never will.** The engine is
separate from the voices it plays. You bring your own.

That is a deliberate choice, and it has three benefits: the repository stays
small enough to clone in seconds, you are never bound by someone else's model
licence, and you can use any voice you like rather than the one we happened to
pick.

## Getting models

StreamAudio uses two kinds, and both are freely available online.

**Piper (TTS)** — the base voice that turns text into speech. Hundreds of voices
across dozens of languages are published as `.onnx` + `.onnx.json` pairs:

- Rhasspy Piper voices — the standard collection, browsable by language and
  quality tier (`x_low`, `low`, `medium`, `high`)

**RVC (voice conversion, optional)** — transforms the Piper output into a
different voice. Takes an ONNX-exported RVC model. Optional: StreamAudio works
fine on Piper alone.

## Installing one

Drop it into `models/` and point `settings.json` at it:

```
models/
  my-voice/
    en_US-something-medium.onnx
    en_US-something-medium.onnx.json
```

Nothing under `models/` is tracked by git. Add as many as you like.

## Licensing — yours to check, not ours

Because you supply the model, its licence is between you and whoever published
it. StreamAudio's licence covers the engine only.

Worth a glance before using a voice commercially:

- **The model licence.** Many published voices are permissive; some restrict
  commercial use or ask for credit to the trainer.
- **The training data.** Whose recordings trained it. This carries rights
  independently of who built the model.

Good models document this themselves — Piper voices usually ship a `MODEL_CARD`
naming the dataset and trainer. RVC's `params.json` has `credit` and
`terms_of_use_url` fields, and they are almost always left empty, so if you
train or download one, filling those in saves you re-deriving it later.

---

*StreamAudio — created and maintained by D-Net Lab. https://lab.dnet.live*
