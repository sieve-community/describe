# Describe

Describe is an app that generates customizable audiovisual descriptions of videos. It uses a combination of visual language models (VLMs) and language models (LMs) to generate a summary of the video content. The app is designed to be highly customizable, allowing users to control the level of visual detail, conciseness, and the influence of spoken context on the final summary.

Functions used:

- [Speech Transcriber](https://www.sievedata.com/functions/sieve/speech_transcriber) for transcriptions
- [MoonDream](https://www.sievedata.com/functions/sieve/moondream) (Low visual detail but fastest)
- [InternLM-X](https://www.sievedata.com/functions/sieve/internlmx-composer-2q) (Medium visual detail and speed)
- [CogVLM](https://www.sievedata.com/functions/sieve/cogvlm-chat) (High visual detail but slowest)

Summaries are generated using GPT-3.5 and GPT-4 by OpenAI. GPT-3.5 is used to combine visual information across a single "chunk" (1 minute of content) as well as to generate a brief transcriptional summary of that chunk. Then, GPT-4 is used to combine all of the information across the chunks to create the final summary.

In order to use the app in your account, you will need to add an `OPENAI_API_KEY` secret in your Sieve account [settings](https://www.sievedata.com/dashboard/settings/secrets).

## Presets and their influence

The app contains 3 presets for `conciseness` and `visual_detail` as well as a boolean for `spoken_context`.

### Visual Detail

This preset influences how much visual detail is present in the final summary by using different VLMs depending on the preset.

- `low` uses [MoonDream](https://www.sievedata.com/functions/sieve/moondream) - a lightweight and very fast VLM that is ideal for scenarios that don't require too much visual information. This is the fastest in terms of response time but not very high in detail
- `medium` uses [InternLM-X](https://www.sievedata.com/functions/sieve/internlmx-composer-2q) - a SOTA VLM that outperforms GPT4v in many benchmarks. This is the 4bit quantized version and falls in between the other two in terms of speed and detail.
- `high` uses [CogVLM](https://www.sievedata.com/functions/sieve/cogvlm-chat) - a very rich and detailed VLM that uses a full sized LLM with it to generate captions, making it capable of highly detailed captions. This is the slowest in terms of response time but high in detail.

The default is `medium`.

### Conciseness

This preset influences how verbose or concise the final summary should be by prompting GPT3.5 and GPT4 with constraints on things like length and finer detail.

- `concise` mode tries to keep the summary to ~100 words and is the slimmest in terms of the output
- `medium` mode tries to keep the summary to ~250 words
- `detailed` mode tries to keep the summary to at most ~500 words and is the most verbose

The default is  `concise`.

### Spoken Context

This option dictates the usage of the speech in the video to influence the final summary. If set to `False`, the summary will be generated based on only the visuals seen across the video. If set to `True`, GPT3.5 will be used first to create shorter summaries which will then be combined at the end to create the final summary.