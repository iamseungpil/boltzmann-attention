# census qwen25_32b -> tb_lodo_mm_32b (n=498, dep=resource)

## aggregate
A: {"parse": 1.0, "n_nodes": 2.6546184738955825, "valid_frac": 1.0, "ntag": 1.6566265060240963, "nself": 0.0, "ndangle": 0.0, "node_f1": 0.8665368767778414, "edge_f1": 0.6388970021500141, "links_ok": null, "argdict_frac": null}
B: {"parse": 1.0, "n_nodes": 2.5281124497991967, "valid_frac": 0.9522088353413657, "ntag": 1.5582329317269077, "nself": 0.018072289156626505, "ndangle": 0.010040160642570281, "node_f1": 0.8688844288241883, "edge_f1": 0.6902929177025564, "links_ok": null, "argdict_frac": null}

## improved: 69 (13.9%)  types={'single': 46, 'chain': 20, 'dag': 3}
A: {"parse": 1.0, "n_nodes": 2.608695652173913, "valid_frac": 1.0, "ntag": 1.5797101449275361, "nself": 0.0, "ndangle": 0.0, "node_f1": 0.6446734424995292, "edge_f1": 0.127536231884058, "links_ok": null, "argdict_frac": null}
B: {"parse": 1.0, "n_nodes": 1.9275362318840579, "valid_frac": 1.0, "ntag": 0.9710144927536232, "nself": 0.0, "ndangle": 0.0, "node_f1": 0.9606740280653323, "edge_f1": 0.952357948010122, "links_ok": null, "argdict_frac": null}

## worsened: 52 (10.4%)  types={'dag': 11, 'chain': 38, 'single': 3}
A: {"parse": 1.0, "n_nodes": 3.826923076923077, "valid_frac": 1.0, "ntag": 2.826923076923077, "nself": 0.0, "ndangle": 0.0, "node_f1": 0.900311120503428, "edge_f1": 0.828511766011766, "links_ok": null, "argdict_frac": null}
B: {"parse": 1.0, "n_nodes": 3.769230769230769, "valid_frac": 0.7080128205128204, "ntag": 2.8846153846153846, "nself": 0.15384615384615385, "ndangle": 0.07692307692307693, "node_f1": 0.5755322668784209, "edge_f1": 0.2239621489621489, "links_ok": null, "argdict_frac": null}

## same: 377 (75.7%)  types={'chain': 190, 'single': 158, 'dag': 29}
A: {"parse": 1.0, "n_nodes": 2.5013262599469495, "valid_frac": 1.0, "ntag": 1.509283819628647, "nself": 0.0, "ndangle": 0.0, "node_f1": 0.9024846653493872, "edge_f1": 0.7063344701275734, "links_ok": null, "argdict_frac": null}
B: {"parse": 1.0, "n_nodes": 2.46684350132626, "valid_frac": 0.9771441202475687, "ntag": 1.4827586206896552, "nself": 0.002652519893899204, "ndangle": 0.002652519893899204, "node_f1": 0.8925471080643496, "edge_f1": 0.7066502463054184, "links_ok": null, "argdict_frac": null}

## examples: improved
### id=10380769 type=single edge 0.00->1.00
GOLD : [{"task": "Image Search", "arguments": ["captivating sunset over mountains landscape"]}]
A    : [{"task": "Image Search", "arguments": ["sunset over mountains"]}, {"task": "Image Downloader", "arguments": ["<node-0>"]}]
B    : [{"task": "Image Search", "arguments": ["sunset over mountains"]}]

### id=10868220 type=chain edge 0.80->1.00
GOLD : [{"task": "URL Extractor", "arguments": ["Let's use this audio clip https://www.example.com/audioclip as background sound in our project video"]}, {"task": "Audio Downloader", "arguments": ["<node-0>"]}, {"task": "Video Synchronization", "arguments": ["<node-1>", "example.mp4"]}, {"task": "Video-to-Audio", "arguments": ["<node-2>"]}]
A    : [{"task": "Audio Downloader", "arguments": ["https://www.example.com/audiofile"]}, {"task": "Video Synchronization", "arguments": ["example.mp4", "<node-0>"]}, {"task": "Video-to-Audio", "arguments": ["<node-1>"]}]
B    : [{"task": "URL Extractor", "arguments": ["Please download audio from https://www.example.com/audiofile and synchronize it."]}, {"task": "Audio Downloader", "arguments": ["<node-0>"]}, {"task": "Video Synchronization", "arguments": ["example.mp4", "<node-1>"]}, {"task": "Video-to-Audio", "arguments": ["<node-2>"]}]

### id=11200164 type=single edge 0.00->1.00
GOLD : [{"task": "Text Search", "arguments": ["top programming languages to learn"]}]
A    : [{"task": "Text Search", "arguments": ["best programming languages"]}, {"task": "Text Summarizer", "arguments": ["<node-0>"]}]
B    : [{"task": "Text Search", "arguments": ["top-rated programming languages"]}]

### id=11656312 type=single edge 0.00->1.00
GOLD : [{"task": "Video Search", "arguments": ["beginner-friendly chocolate chip cookies baking tutorial"]}]
A    : [{"task": "Video Search", "arguments": ["how to bake chocolate chip cookies"]}, {"task": "Video Downloader", "arguments": ["<node-0>"]}]
B    : [{"task": "Video Search", "arguments": ["baking chocolate chip cookies"]}]

### id=12219157 type=chain edge 0.00->0.25
GOLD : [{"task": "Text Paraphraser", "arguments": ["Once upon a time, in a kingdom far away, there lived a young princess named Example. She was beautiful, kind, and loved by all her subjects. One day, she received an example.png from a mysterious stranger, which led her on an incredible journey. Along the way, she discovered a secret cave filled with magical example.jpg and met a talking example.mp4, who became her loyal companion. Together, they overcame various challenges and eventually found a long-lost example.wav that would transform their lives forever."]}, {"task": "Text Summarizer", "argumen
A    : [{"task": "Text Summarizer", "arguments": ["Once upon a time, in a kingdom far away, there lived a young princess named Example. She was beautiful, kind, and loved by all her subjects. One day, she received an example.png from a mysterious stranger, which led her on an incredible journey. Along the way, she discovered a secret cave filled with magical example.jpg and met a talking example.mp4, who became her loyal companion. Together, they overcame various challenges and eventually found a long-lost example.wav that would transform their lives forever."]}, {"task": "Text-to-Image", "arguments"
B    : [{"task": "Text Summarizer", "arguments": ["Once upon a time, in a kingdom far away, there lived a young princess named Example. She was beautiful, kind, and loved by all her subjects. One day, she received an example.png from a mysterious stranger, which led her on an incredible journey. Along the way, she discovered a secret cave filled with magical example.jpg and met a talking example.mp4, who became her loyal companion. Together, they overcame various challenges and eventually found a long-lost example.wav that would transform their lives forever."]}, {"task": "Text-to-Video", "arguments"

### id=12966686 type=single edge 0.00->1.00
GOLD : [{"task": "Video-to-Text", "arguments": ["example.mp4"]}]
A    : [{"task": "Video-to-Audio", "arguments": ["example.mp4"]}, {"task": "Audio-to-Text", "arguments": ["<node-0>"]}]
B    : [{"task": "Video-to-Text", "arguments": ["example.mp4"]}]

## examples: worsened
### id=11545630 type=dag edge 1.00->0.40
GOLD : [{"task": "Audio Downloader", "arguments": ["example.wav"]}, {"task": "Audio Effects", "arguments": ["<node-0>", "<node-2>"]}, {"task": "Text Search", "arguments": ["<node-3>"]}, {"task": "Video-to-Text", "arguments": ["example.mp4"]}]
A    : [{"task": "Audio Downloader", "arguments": ["example.wav"]}, {"task": "Video-to-Text", "arguments": ["example.mp4"]}, {"task": "Text Search", "arguments": ["<node-1>"]}, {"task": "Audio Effects", "arguments": ["<node-0>", "<node-2>"]}]
B    : [{"task": "Audio Downloader", "arguments": ["example.wav"]}, {"task": "Video-to-Text", "arguments": ["example.mp4"]}, {"task": "Audio Effects", "arguments": ["<node-0>", "<node-1>"]}]

### id=11831430 type=chain edge 0.50->0.00
GOLD : [{"task": "Text Search", "arguments": ["Solar energy conversion process"]}, {"task": "Text Simplifier", "arguments": ["<node-0>"]}, {"task": "Text to Voice Conversion", "arguments": ["<node-1>", "female"]}]
A    : [{"task": "Text Search", "arguments": ["how solar energy works"]}, {"task": "Text Simplifier", "arguments": ["<node-0>"]}, {"task": "Text-to-Audio", "arguments": ["<node-1>", "female"]}]
B    : [{"task": "Text Simplifier", "arguments": ["<node-1>"]}, {"task": "Text-to-Audio", "arguments": ["<node-0>"]}, {"task": "Voice Changer", "arguments": ["<node-2>", "female"]}, {"task": "Web Search", "arguments": ["how solar energy works"]}]

### id=13468023 type=chain edge 1.00->0.00
GOLD : [{"task": "Text Downloader", "arguments": ["www.exampleblog.com/article"]}, {"task": "Keyword Extractor", "arguments": ["<node-0>"]}, {"task": "Voice Changer", "arguments": ["example.wav", "<node-1>"]}]
A    : [{"task": "Text Downloader", "arguments": ["www.exampleblog.com/article"]}, {"task": "Keyword Extractor", "arguments": ["<node-0>"]}, {"task": "Voice Changer", "arguments": ["example.wav", "<node-1>"]}]
B    : [{"task": "Text Keyword Extraction", "arguments": ["www.exampleblog.com/article"]}, {"task": "Voice Modification", "arguments": ["<node-0>", "example.wav"]}]

### id=13543080 type=dag edge 0.80->0.33
GOLD : [{"task": "Video Stabilizer", "arguments": ["example.mp4"]}, {"task": "Voice Changer", "arguments": ["example.wav", "Change tone, pitch, and gender as needed."]}, {"task": "Video Synchronization", "arguments": ["<node-0>", "<node-1>"]}, {"task": "Video-to-Text", "arguments": ["<node-2>"]}]
A    : [{"task": "Voice Changer", "arguments": ["example.wav", "modify tone, pitch, and gender"]}, {"task": "Video Stabilizer", "arguments": ["example.mp4"]}, {"task": "Video Synchronization", "arguments": ["<node-1>", "<node-0>"]}, {"task": "Audio-to-Text", "arguments": ["example.wav"]}]
B    : [{"task": "Voice Changer", "arguments": ["example.wav", "theme-specific adjustments"]}, {"task": "Video Synchronization", "arguments": ["example.mp4", "<node-0>"]}, {"task": "Video Stabilizer", "arguments": ["<node-1>"]}, {"task": "Video-to-Text", "arguments": ["<node-2>"]}]

### id=13864445 type=chain edge 1.00->0.00
GOLD : [{"task": "Text Sentiment Analysis", "arguments": ["I am extremely happy with the results of my work."]}, {"task": "Text-to-Video", "arguments": ["<node-0>"]}, {"task": "Video-to-Audio", "arguments": ["<node-1>"]}]
A    : [{"task": "Text Sentiment Analysis", "arguments": ["I am extremely happy with the results of my work."]}, {"task": "Text-to-Video", "arguments": ["<node-0>"]}, {"task": "Video-to-Audio", "arguments": ["<node-1>"]}]
B    : [{"task": "Sentiment Analysis", "arguments": ["I am extremely happy with the results of my work."]}, {"task": "Video Creation", "arguments": ["<node-0>"]}, {"task": "Audio Extraction", "arguments": ["<node-1>"]}]

### id=13921012 type=dag edge 0.67->0.00
GOLD : [{"task": "Text Summarizer", "arguments": ["Convert the hard to understand, deep and slow speech of a male into a higher-pitched, faster speech of a female."]}, {"task": "Video-to-Audio", "arguments": ["example.mp4"]}, {"task": "Voice Changer", "arguments": ["<node-1>", "<node-0>"]}]
A    : [{"task": "Video-to-Audio", "arguments": ["example.mp4"]}, {"task": "Voice Changer", "arguments": ["<node-0>", "Change to a higher-pitched, faster-speaking female voice."]}]
B    : [{"task": "Audio Extraction", "arguments": ["example.mp4"]}, {"task": "Voice Changer", "arguments": ["<node-0>", "A male speaker with a deep voice and a slow pace of speech is hard to understand. Please change to a higher-pitched, faster-speaking female voice."]}]

