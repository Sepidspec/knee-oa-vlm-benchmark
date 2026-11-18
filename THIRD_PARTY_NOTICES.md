# Source and third-party materials

- Dataset: Shashwat Tiwari, *Knee Osteoarthritis Dataset with Severity Grading*,
  [Kaggle source](https://www.kaggle.com/datasets/shashwatwork/knee-osteoarthritis-dataset-with-severity).
  Download the radiographs separately and check the source's current access and
  reuse terms. The radiographs are not included here. The completed run records
  source filenames, hashes, folder labels and provisional grouping; clinical
  participant linkage was not independently established.
- Base model: [Qwen2.5-VL-3B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct),
  revision `66285546d2b821cf421d4f5eb2576359d3770cd3`.
  The model snapshot's Qwen Research License and model card are preserved as
  `protocol/model_LICENSE` and `protocol/model_README.md` inside the analysis
  ZIP. Tokenizer/processor metadata in that archive remain subject to their
  original terms. Pretrained model and adapted weight files are not included.
- The ImageNet-initialized ResNet-18 is loaded through torchvision; see the
  [torchvision model documentation](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.resnet18.html).
- Dependencies (PyTorch, torchvision, Transformers, PEFT, bitsandbytes, NumPy,
  pandas, SciPy, Matplotlib and Pillow) retain their respective licenses.

The repository does not imply endorsement by any data or model provider.
