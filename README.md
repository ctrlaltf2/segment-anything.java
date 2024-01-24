# segment-anything.java
Meta's [Segment Anything (SAM)](https://github.com/facebookresearch/segment-anything) model, ported to Java SE 17.

Eventual goal is to use this in a [JOSM](https://josm.openstreetmap.de/) [plugin](https://github.com/JOSM/josm-plugins) for segmentation of aerial imagery for OpenStreetMap. Further versions from that will be [finetuned](https://github.com/ctrlaltf2/segment-any-landuse) on aerial imagery for better results.

## Roadmap
 - [x] Reproduce ONNX export of encoder and decoder (ref: https://github.com/visheratin/segment-anything)
 - [x] Image loading and preprocessing
 - [x] Encoder forward pass
 - [x] OnnxTensor to primitive matrix type conversion
 - [x] Decoder forward pass, basic coordinate-based prompt
 - [ ] Decoder mask post-processing (mapping back to the input image)
 - [ ] Multi-mask support (currently T/F value is baked into the ONNX model at export-time)
 - [ ] Error handling TODOs
 - [ ] Improve upon ONNX export method
 - [ ] Figure out a place to host the ONNX models (~5 GB total)
 - [ ] FastSAM? It's such a different model though architecturally, might be out of the scope of this repository.
 - [ ] GPU runtimes?
