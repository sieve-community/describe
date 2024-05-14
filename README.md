# Describe

Describe is an app that generates customizable audiovisual descriptions of videos. It uses a combination of visual language models (VLMs) and language models (LMs) to generate a summary of the video content. The app is designed to be highly customizable, allowing users to control the level of visual detail, conciseness, and the influence of spoken context on the final summary.

## Quick Demo

To run this on your own videos quickly, check out the app on Sieve [here](https://www.sievedata.com/functions/sieve/describe). Below is a quick example output of an interview with Boris Johnson:

> In this video, a blonde man in a blue Xchanging shirt, accompanied by another person in similar attire, engages with the press outdoors. As the reporters incessantly question him about his regrets regarding earlier comments, he deflects by offering tea, emphasizing the gesture's humanitarian nature and expressing sympathy for the reporters' long wait. Despite repeated inquiries, he maintains a polite demeanor, focusing on the act of offering tea rather than providing direct answers. The visual content captures this exchange against an outdoor backdrop, highlighting the interaction's dynamic and the man's approach in handling the situation.

[![Example](https://storage.googleapis.com/sieve-public-data/boris-preview.png)](https://www.sievedata.com/functions/sieve/describe)

Note: This app uses functions that run on the [Sieve](https://www.sievedata.com) platform. You will need a Sieve account to use this. However, all the logic is open-source and pretty simple to replicate in your own environment as well. The functions used are:

- [Speech Transcriber](https://www.sievedata.com/functions/sieve/speech_transcriber) for transcriptions
- [MoonDream](https://www.sievedata.com/functions/sieve/moondream) (Low visual detail but fastest)
- [InternLM-X](https://www.sievedata.com/functions/sieve/internlmx-composer-2q) (Medium visual detail and speed)
- [CogVLM](https://www.sievedata.com/functions/sieve/cogvlm-chat) (High visual detail but slower)
- [VILA](https://www.sievedata.com/functions/sieve/vila) (Ultra high visual detail but slowest)

Summaries are generated using an LLM backend that combines various pieces of context. We have two options for this: `openai` and `mixtral`.
- If `openai` is selected, we use GPT-4 by OpenAI.
- If `mixtral` is selected, we use the `mistralai/Mixtral-8x7B-Instruct-v0.1` model hosted on Together AI.

In order to use the app in your account, you will need to add an `OPENAI_API_KEY` or a `TOGETHERAI_API_KEY` secret in your Sieve account [settings](https://www.sievedata.com/dashboard/settings/secrets) and specify the backend accordingly.

## Presets and their influence

The app contains 3 presets for `conciseness` and `visual_detail` as well as a boolean for `spoken_context`.

### Visual Detail

This preset influences how much visual detail is present in the final summary by using different VLMs depending on the preset.

- `low` uses [MoonDream](https://www.sievedata.com/functions/sieve/moondream) - a lightweight and very fast VLM that is ideal for scenarios that don't require too much visual information. This is the fastest in terms of response time but not very high in detail
- `medium` uses [InternLM-X](https://www.sievedata.com/functions/sieve/internlmx-composer-2q) - a SOTA VLM that outperforms GPT4v in many benchmarks. This is the 4bit quantized version and falls in between the other two in terms of speed and detail.
- `high` uses [CogVLM](https://www.sievedata.com/functions/sieve/cogvlm-chat) - a very rich and detailed VLM that uses a full sized LLM with it to generate captions, making it capable of highly detailed captions. This is slower in terms of response time but high in detail.
- `ultra` uses [VILA](https://www.sievedata.com/functions/sieve/vila) - a VLM pretrained with video understanding as a core focus, allowing it to work with multiple frames of a video at once. This allows the VLM to work with longer sections of a video at the same time, capturing more detail and providing better captions.

The default is `high`.

### Conciseness

This preset influences how verbose or concise the final summary should be by prompting GPT3.5 and GPT4 with constraints on things like length and finer detail.

- `concise` mode tries to keep the summary to 4 sentences and is the slimmest in terms of the output
- `medium` mode tries to keep the summary to 8 sentences
- `detailed` mode tries to keep the summary to at most 12 sentences and is the most verbose

The default is  `concise`.

### Spoken Context

This option dictates the usage of the speech in the video to influence the final summary. If set to `False`, the summary will be generated based on only the visuals seen across the video. If set to `True`, GPT3.5 will be used first to create shorter summaries which will then be combined at the end to create the final summary.

### Enable References

This option will give sentence level references for each sentence in the summary. It gives references to audio transcripts, visual captions and objects in the scenes, with start time and end_time indicating the span of each sentence in the video. It enables chunk_by_scene by default for better context. 
