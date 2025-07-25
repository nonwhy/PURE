# PURE Training

For efficiency considerations, the multi-modal datasets are pre-tokenized into sequences of token ids. This leads to significantly faster training

## Pre-tokenization


### 1. Run Tokenization

This stage tokenizes each data point, consisting of interleaved image and text, into a single sequence of integer tokens. After tokenization, the sequence is saved to disk for trainining-time usage. Together with the saved tokens, a json-formatted record file is also generated for indexing all the saved token files. For faster tokenization, you may use multiple GPUs and dispatch different subsets of data to them.

#### Command:

```bash
for i in {0..7}
do
  export CUDA_VISIBLE_DEVICES=${i}
  python -u pre_tokenize/pre_tokenize.py \
  --splits=8 \
  --rank=${i} \
  --in_filename /path/to/in_filename.json \
  --out_dir /path/to/out_dir \
  --target_size 768 &> ${i}.log &
done
```

#### Format of Input File:

`in_filename` is expected to be a json file with the following format:
```python
[
    {...},
    {...},
    {
        "conversations":[
            {
                "from": "human",
                "value": "Your instruction. <|image|>"
            },
            {
                "from": "gpt",
                "value": "Expected text output. <|image|>"
            }
        ],
        "image": ["/path/to/image1.png", "path/to/image2.png"]
    },
    {...},
    {...}
]
```

*Rules:*

1. The file is a list of dictionaries, and each dictionary represents a data point
2. Each dictionary contains the key "conversations"
3. If the conversation involves image(s), the data point should also contain the `image` key, otherwise the `image` key can be omitted
4. The location of each image should be explicitly specified in the conversation using the `<|image|>` symbol
   1. Apparently, the number of occurrences of the `<|image|>` symbol should be equal to the number of images in the `image` key


#### How to adapt to your own format:

If you have your own data with a different format, you can easily adapt the code to deal with it by modifying the `pre_tokenize.py` file.
We have prepared the space, which is in `ItemProcessor.process_item`, for adding your logic that converts data points of your own format into the standard format.

### 2. Concat Records

After tokenization, You need to concat the record files generated by different processes (GPUs) into one single record file.
**Note that we use the term "record file" to refer to the meta file that contains the information of all the saved token files,
which is different from the token files themselves.**

```bash
python -u pre_tokenize/concat_record.py \
--sub_record_dir /path/to/out_dir \
--save_path /path/to/out_dir/record.json
```

## Training

#### Command:
We provide an example experiment scripts [scripts/train.sh](scripts/train.sh) for training the 7B model. You can run the following command to start training:

```bash
./exps/7B.sh
```

#### About the `--data_config` argument:
The ``--data_config`` argument should point to a `*.yaml` file, which is a meta file that gathers one or multiple record files.
In other words, you may pre-tokenize multiple datasets independently and list the record files in the same data config file for joint training.