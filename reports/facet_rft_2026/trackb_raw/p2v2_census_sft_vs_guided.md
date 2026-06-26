# census tb_lodo_mm_32b -> tb_lodo_mm_32b_guided (n=499, dep=resource)

## aggregate
A: {"parse": 1.0, "n_nodes": 2.531062124248497, "valid_frac": 0.9513026052104211, "ntag": 1.5611222444889779, "nself": 0.018036072144288578, "ndangle": 0.01002004008016032, "node_f1": 0.8671431774638191, "edge_f1": 0.6889095651620702, "links_ok": null, "argdict_frac": null}
B: {"parse": 1.0, "n_nodes": 2.5370741482965933, "valid_frac": 1.0, "ntag": 1.5671342685370742, "nself": 0.02004008016032064, "ndangle": 0.02004008016032064, "node_f1": 0.9022602202962928, "edge_f1": 0.7094363329834273, "links_ok": null, "argdict_frac": null}

## improved: 21 (4.2%)  types={'chain': 17, 'dag': 4}
A: {"parse": 1.0, "n_nodes": 3.761904761904762, "valid_frac": 0.40396825396825403, "ntag": 2.9047619047619047, "nself": 0.047619047619047616, "ndangle": 0.0, "node_f1": 0.3650024578596008, "edge_f1": 0.15767195767195769, "links_ok": null, "argdict_frac": null}
B: {"parse": 1.0, "n_nodes": 3.857142857142857, "valid_frac": 1.0, "ntag": 3.0476190476190474, "nself": 0.047619047619047616, "ndangle": 0.2857142857142857, "node_f1": 0.9119562976705834, "edge_f1": 0.7496598639455784, "links_ok": null, "argdict_frac": null}

## worsened: 4 (0.8%)  types={'chain': 4}
A: {"parse": 1.0, "n_nodes": 3.5, "valid_frac": 0.9375, "ntag": 2.25, "nself": 0.0, "ndangle": 0.0, "node_f1": 0.8041666666666667, "edge_f1": 0.6, "links_ok": null, "argdict_frac": null}
B: {"parse": 1.0, "n_nodes": 3.75, "valid_frac": 1.0, "ntag": 2.5, "nself": 0.0, "ndangle": 0.0, "node_f1": 0.7708333333333333, "edge_f1": 0.08333333333333333, "links_ok": null, "argdict_frac": null}

## same: 474 (95.0%)  types={'chain': 227, 'single': 207, 'dag': 40}
A: {"parse": 1.0, "n_nodes": 2.4683544303797467, "valid_frac": 0.9756680731364277, "ntag": 1.4957805907172996, "nself": 0.016877637130801686, "ndangle": 0.010548523206751054, "node_f1": 0.8899213655542773, "edge_f1": 0.7131957002210166, "links_ok": null, "argdict_frac": null}
B: {"parse": 1.0, "n_nodes": 2.4683544303797467, "valid_frac": 1.0, "ntag": 1.4936708860759493, "nself": 0.0189873417721519, "ndangle": 0.008438818565400843, "node_f1": 0.9029397349017607, "edge_f1": 0.7129378474315183, "links_ok": null, "argdict_frac": null}

## examples: improved
### id=13468023 type=chain edge 0.00->0.67
GOLD : [{"task": "Text Downloader", "arguments": ["www.exampleblog.com/article"]}, {"task": "Keyword Extractor", "arguments": ["<node-0>"]}, {"task": "Voice Changer", "arguments": ["example.wav", "<node-1>"]}]
A    : [{"task": "Text Keyword Extraction", "arguments": ["www.exampleblog.com/article"]}, {"task": "Voice Modification", "arguments": ["<node-0>", "example.wav"]}]
B    : [{"task": "Keyword Extractor", "arguments": ["www.exampleblog.com/article"]}, {"task": "Voice Changer", "arguments": ["example.wav", "<node-0>"]}]

### id=13864445 type=chain edge 0.00->1.00
GOLD : [{"task": "Text Sentiment Analysis", "arguments": ["I am extremely happy with the results of my work."]}, {"task": "Text-to-Video", "arguments": ["<node-0>"]}, {"task": "Video-to-Audio", "arguments": ["<node-1>"]}]
A    : [{"task": "Sentiment Analysis", "arguments": ["I am extremely happy with the results of my work."]}, {"task": "Video Creation", "arguments": ["<node-0>"]}, {"task": "Audio Extraction", "arguments": ["<node-1>"]}]
B    : [{"task": "Text Sentiment Analysis", "arguments": ["I am extremely happy with the results of my work."]}, {"task": "Text-to-Video", "arguments": ["<node-0>"]}, {"task": "Video-to-Audio", "arguments": ["<node-1>"]}]

### id=13921012 type=dag edge 0.00->0.67
GOLD : [{"task": "Text Summarizer", "arguments": ["Convert the hard to understand, deep and slow speech of a male into a higher-pitched, faster speech of a female."]}, {"task": "Video-to-Audio", "arguments": ["example.mp4"]}, {"task": "Voice Changer", "arguments": ["<node-1>", "<node-0>"]}]
A    : [{"task": "Audio Extraction", "arguments": ["example.mp4"]}, {"task": "Voice Changer", "arguments": ["<node-0>", "A male speaker with a deep voice and a slow pace of speech is hard to understand. Please change to a higher-pitched, faster-speaking female voice."]}]
B    : [{"task": "Video-to-Audio", "arguments": ["example.mp4"]}, {"task": "Voice Changer", "arguments": ["<node-0>", "A male speaker with a deep voice and a slow pace of speech is hard to understand. Please change to a higher-pitched, faster-speaking female voice."]}]

### id=17603966 type=dag edge 0.80->1.00
GOLD : [{"task": "Video-to-Text", "arguments": ["example.mp4"]}, {"task": "Keyword Extractor", "arguments": ["<node-0>"]}, {"task": "URL Extractor", "arguments": ["<node-0>"]}, {"task": "Text Grammar Checker", "arguments": ["<node-1>"]}]
A    : [{"task": "Video-to-Text", "arguments": ["example.mp4"]}, {"task": "Keyword Extractor", "arguments": ["<node-0>"]}, {"task": "Text Grammar Checker", "arguments": ["<node-1>"]}]
B    : [{"task": "Video-to-Text", "arguments": ["example.mp4"]}, {"task": "Keyword Extractor", "arguments": ["<node-0>"]}, {"task": "URL Extractor", "arguments": ["<node-0>"]}, {"task": "Text Grammar Checker", "arguments": ["<node-1>"]}]

### id=18824625 type=chain edge 0.50->1.00
GOLD : [{"task": "Text Sentiment Analysis", "arguments": ["I am really loving this new phone I got. The camera quality is exceptional."]}, {"task": "Text-to-Image", "arguments": ["<node-0>"]}, {"task": "Image-to-Text", "arguments": ["<node-1>"]}, {"task": "Text Grammar Checker", "arguments": ["<node-2>"]}, {"task": "Keyword Extractor", "arguments": ["<node-3>"]}]
A    : [{"task": "Text Sentiment Analysis", "arguments": ["I am really loving this new phone I got. The camera quality is exceptional."]}, {"task": "Sentiment-to-Image", "arguments": ["<node-0>"]}, {"task": "Image-to-Text", "arguments": ["<node-1>"]}, {"task": "Text Grammar Checker", "arguments": ["<node-2>"]}, {"task": "Keyword Extractor", "arguments": ["<node-3>"]}]
B    : [{"task": "Text Sentiment Analysis", "arguments": ["I am really loving this new phone I got. The camera quality is exceptional."]}, {"task": "Text-to-Image", "arguments": ["<node-0>"]}, {"task": "Image-to-Text", "arguments": ["<node-1>"]}, {"task": "Text Grammar Checker", "arguments": ["<node-2>"]}, {"task": "Keyword Extractor", "arguments": ["<node-3>"]}]

### id=20648392 type=chain edge 0.00->0.67
GOLD : [{"task": "Video-to-Text", "arguments": ["example.mp4"]}, {"task": "Video Search", "arguments": ["<node-3>"]}, {"task": "Video Synchronization", "arguments": ["<node-1>", "example.wav"]}, {"task": "Video-to-Image", "arguments": ["<node-2>"]}]
A    : [{"task": "Video Content Analysis", "arguments": ["example.mp4"]}, {"task": "Video Search", "arguments": ["<node-0>"]}, {"task": "Video Audio Synchronization", "arguments": ["<node-1>", "example.wav"]}, {"task": "Video Image Extraction", "arguments": ["<node-2>"]}]
B    : [{"task": "Video-to-Text", "arguments": ["example.mp4"]}, {"task": "Video Search", "arguments": ["<node-0>"]}, {"task": "Video Synchronization", "arguments": ["<node-1>", "example.wav"]}, {"task": "Video-to-Image", "arguments": ["<node-2>"]}]

## examples: worsened
### id=19565758 type=chain edge 0.40->0.00
GOLD : [{"task": "Text Grammar Checker", "arguments": ["<node-2>"]}, {"task": "Keyword Extractor", "arguments": ["<node-2>"]}, {"task": "Image Search", "arguments": ["<node-1>"]}]
A    : [{"task": "Text Grammar Checker", "arguments": ["blog_post_text"]}, {"task": "Keyword Extractor", "arguments": ["<node-0>"]}, {"task": "Image Search", "arguments": ["<node-1>", "example.jpg"]}]
B    : [{"task": "Text Grammar Checker", "arguments": ["blog_post_text"]}, {"task": "Topic Generator", "arguments": ["<node-0>"]}, {"task": "Image Search", "arguments": ["<node-1>"]}]

### id=21366352 type=chain edge 0.67->0.33
GOLD : [{"task": "Audio Downloader", "arguments": ["https://example.com/audio.wav"]}, {"task": "Audio Noise Reduction", "arguments": ["<node-0>"]}, {"task": "Video Synchronization", "arguments": ["example.mp4", "<node-1>"]}, {"task": "Video-to-Text", "arguments": ["<node-2>"]}]
A    : [{"task": "Audio Noise Reduction", "arguments": ["<node-1>"]}, {"task": "Audio Downloader", "arguments": ["https://example.com/audio.wav"]}, {"task": "Video-to-Audio", "arguments": ["example.mp4"]}, {"task": "Video Synchronization", "arguments": ["example.mp4", "<node-0>"]}, {"task": "Audio-to-Text", "arguments": ["<node-3>"]}]
B    : [{"task": "Audio Noise Reduction", "arguments": ["<node-2>"]}, {"task": "Audio Downloader", "arguments": ["https://example.com/audio.wav"]}, {"task": "Video-to-Audio", "arguments": ["example.mp4"]}, {"task": "Video Synchronization", "arguments": ["example.mp4", "<node-0>"]}, {"task": "Audio-to-Text", "arguments": ["<node-3>"]}]

### id=29582096 type=chain edge 0.67->0.00
GOLD : [{"task": "Text Paraphraser", "arguments": ["Despite the plethora of challenges that Rachel encountered during her excruciatingly arduous journey, she remained resolute in her determination to reach her ultimate goal."]}, {"task": "Text Simplifier", "arguments": ["<node-0>"]}, {"task": "Text Grammar Checker", "arguments": ["<node-1>"]}]
A    : [{"task": "Text Simplifier", "arguments": ["Despite the plethora of challenges that Rachel encountered during her excruciatingly arduous journey, she remained resolute in her determination to reach her ultimate goal."]}, {"task": "Text Grammar Checker", "arguments": ["<node-0>"]}]
B    : [{"task": "Text Simplifier", "arguments": ["Despite the plethora of challenges that Rachel encountered during her excruciatingly arduous journey, she remained resolute in her determination to reach her ultimate goal."]}, {"task": "Text Paraphraser", "arguments": ["<node-0>"]}, {"task": "Text Grammar Checker", "arguments": ["<node-1>"]}]

### id=30112536 type=chain edge 0.67->0.00
GOLD : [{"task": "Video-to-Audio", "arguments": ["example.mp4"]}, {"task": "Audio Noise Reduction", "arguments": ["<node-0>"]}, {"task": "Audio Effects", "arguments": ["<node-1>", "reverb"]}, {"task": "Voice Changer", "arguments": ["<node-2>", "higher pitch, female tone"]}]
A    : [{"task": "Audio Extraction", "arguments": ["example.mp4"]}, {"task": "Audio Noise Reduction", "arguments": ["<node-0>"]}, {"task": "Audio Effects", "arguments": ["<node-1>", "reverb"]}, {"task": "Voice Changer", "arguments": ["<node-2>", "higher pitch, female tone"]}]
B    : [{"task": "Audio Noise Reduction", "arguments": ["<node-3>"]}, {"task": "Audio-to-Text", "arguments": ["<node-0>"]}, {"task": "Video-to-Audio", "arguments": ["example.mp4"]}, {"task": "Voice Changer", "arguments": ["<node-1>", "female"]}]

