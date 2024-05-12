# Crop registration pipeline
The pipeline to compute 3D models of a crop field and to properly register them. Includes the codes of the experiments done in the paper. This relies on the repository [Hierarchical_Localization](https://github.com/cvg/Hierarchical-Localization) for performing localization between query and reference images.

## Installation
```shell
cd interday_crop_registration/crop_alignment
conda env create -f environment.yaml
conda activate hloc
```

## How to run
There are two files provided, each correspond to an experiment in the paper.
* [`run_exp.py`](./run_exp.py): Experiment V.B in the paper. Evaluate the registration pipeline's capability when varying the time intervals between the query and reference models. Query-Reference pairs are defined [here](./query_ref_pair.py).
* [`run_pipeline.py`](./run_pipeline.py): Experiment V.C in the paper. Mimic the real use case. Run the registration pipeline sequentially, with ground truth model only used in refining the first model.

To run them, follow the following steps:

1. Prepare datasets
    - [Download](../README.md#download-datasets) the dataset for crop alignment.
    - Modify the paths in the `run_xxx.py` scripts to match the paths of the extracted downloaded files first. Open [`run_exp.py`](./run_exp.py) or [`run_pipeline.py`](./run_pipeline.py) and change the following path:
        ```shell
        # TODO: Change the path to match the paths of the extracted downloaded dataset
        source_images_path = '/path/to/Wheat_2019_images'    
        ```
2. Invoke retrained models
    - [Option 1] Use the retrained models used in the paper. In that case, you can skip this step. 
    *Note: We provide the two models mentioned in the paper, one trained with considering height change and one without. The name idtifiers are `'loftr_23_0.5_hc'` and `'loftr_23_0.5'`. Here's an example of using them to identify a matcher from [`run_exp.py`](./run_exp.py#L276-L277): 
        ```shell
        extractor_matchers = [
            ['sift', 'NN-ratio'],
            ['superpoint_aachen', 'superglue'],
            [None, 'loftr'],  # extractor for LoFTR is None because it does not need feature extractor
            [None, 'loftr_23_0.5'],  # retrained LoFTR without height change
            [None, 'loftr_23_0.5_hc'],  # retrained LoFTR with height change
        ]
        ```

    - [Option 2] [Retrain](../LoFTR/README.md#how-to-train) a new LoFTR model. Refer to [Hierarchical_Localization](./Hierarchical_Localization/README.md#modifications-to-enable-retrained-loftr-models) for instructions on enabling `hloc` to support self-trained LoFTR models.

3. Run the code
    ```shell
    python run_xxx.py
    ```

