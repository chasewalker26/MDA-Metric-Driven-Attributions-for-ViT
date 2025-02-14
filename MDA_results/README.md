<h1>EVALUATION ON ALL MODELS</h1>

`cd MDA_results/evaluations`

You should have the imagenet ILSVRC2012 validation set of images in a folder to point these tests to.

<h3>Insertion, Deletion, and MAS Insertion and Deletion Tests (Table 1, 4, 5, 7, 8, 9)</h3>

```
python3 MDATest.py --function GC --model VIT_base_16 --image_count 5000 --imagenet <path-to-imagenet-data> --gpu 0
python3 MDATest.py --function IG --model VIT_base_16 --image_count 5000 --imagenet <path-to-imagenet-data> --gpu 0
python3 MDATest.py --function LRP --model VIT_base_16 --image_count 5000 --imagenet <path-to-imagenet-data> --gpu 0
python3 MDATest.py --function VIT_CX --model VIT_base_16 --image_count 5000 --imagenet <path-to-imagenet-data> --gpu 0
python3 MDATest.py --function Bidirectional --model VIT_base_16 --image_count 5000 --imagenet <path-to-imagenet-data> --gpu 0
python3 MDATest.py --function Transition_attn --model VIT_base_16 --image_count 5000 --imagenet <path-to-imagenet-data> --gpu 0
python3 MDATest.py --function TIS --model VIT_base_16 --image_count 5000 --imagenet <path-to-imagenet-data> --gpu 0
python3 MDATest.py --function Calibrate --model VIT_base_16 --image_count 5000 --imagenet <path-to-imagenet-data> --gpu 0


python3 MDATest.py --function GC --model VIT_tiny_16 --image_count 5000 --imagenet <path-to-imagenet-data> --gpu 1
python3 MDATest.py --function IG --model VIT_tiny_16 --image_count 5000 --imagenet <path-to-imagenet-data> --gpu 1
python3 MDATest.py --function LRP --model VIT_tiny_16 --image_count 5000 --imagenet <path-to-imagenet-data> --gpu 1
python3 MDATest.py --function VIT_CX --model VIT_tiny_16 --image_count 5000 --imagenet <path-to-imagenet-data> --gpu 1
python3 MDATest.py --function Bidirectional --model VIT_tiny_16 --image_count 5000 --imagenet <path-to-imagenet-data> --gpu 2
python3 MDATest.py --function Transition_attn --model VIT_tiny_16 --image_count 5000 --imagenet <path-to-imagenet-data> --gpu 2
python3 MDATest.py --function TIS --model VIT_tiny_16 --image_count 5000 --imagenet <path-to-imagenet-data> --gpu 2
python3 MDATest.py --function Calibrate --model VIT_tiny_16 --image_count 5000 --imagenet <path-to-imagenet-data> --gpu 2


python3 MDATest.py --function GC --model VIT_base_32 --image_count 5000 --imagenet <path-to-imagenet-data> --gpu 3
python3 MDATest.py --function IG --model VIT_base_32 --image_count 5000 --imagenet <path-to-imagenet-data> --gpu 3
python3 MDATest.py --function LRP --model VIT_base_32 --image_count 5000 --imagenet <path-to-imagenet-data> --gpu 3
python3 MDATest.py --function VIT_CX --model VIT_base_32 --image_count 5000 --imagenet <path-to-imagenet-data> --gpu 3
python3 MDATest.py --function Bidirectional --model VIT_base_32 --image_count 5000 --imagenet <path-to-imagenet-data> --gpu 3
python3 MDATest.py --function Transition_attn --model VIT_base_32 --image_count 5000 --imagenet <path-to-imagenet-data> --gpu 3
python3 MDATest.py --function TIS --model VIT_tiny_16 --image_count 5000 --imagenet <path-to-imagenet-data> --gpu 2
python3 MDATest.py --function Calibrate --model VIT_tiny_16 --image_count 5000 --imagenet <path-to-imagenet-data> --gpu 2

```

<h3>ImageNet Segmentation Tests (Table 2)</h3>

Point these tests to the following dataset:

http://calvin-vision.net/bigstuff/proj-imagenet/data/gtsegs_ijcv.mat


```
python3 imagenet_seg_eval.py --method GC --imagenet-seg-path <path-to-gtsegs_ijcv.mat> --gpu 0 --model VIT_base_32 --acc_cutoff 10
python3 imagenet_seg_eval.py --method IG --imagenet-seg-path <path-to-gtsegs_ijcv.mat> --gpu 0 --model VIT_base_32 --acc_cutoff 10
python3 imagenet_seg_eval.py --method VIT_CX --imagenet-seg-path <path-to-gtsegs_ijcv.mat> --gpu 0 --model VIT_base_32 --acc_cutoff 10
python3 imagenet_seg_eval.py --method Transition_attn --imagenet-seg-path <path-to-gtsegs_ijcv.mat> --gpu 0 --model VIT_base_32 --acc_cutoff 10
python3 imagenet_seg_eval.py --method LRP --imagenet-seg-path <path-to-gtsegs_ijcv.mat> --gpu 0 --model VIT_base_32 --acc_cutoff 10
python3 imagenet_seg_eval.py --method Bidirectional --imagenet-seg-path <path-to-gtsegs_ijcv.mat> --gpu 0 --model VIT_base_32 --acc_cutoff 10
python3 imagenet_seg_eval.py --method TIS --imagenet-seg-path <path-to-gtsegs_ijcv.mat> --gpu 0 --model VIT_base_32 --acc_cutoff 10
python3 imagenet_seg_eval.py --method Calibrate_Best_Possible --imagenet-seg-path <path-to-gtsegs_ijcv.mat> --gpu 0 --model VIT_base_32 --acc_cutoff 10
python3 imagenet_seg_eval.py --method SHAP --imagenet-seg-path <path-to-gtsegs_ijcv.mat> --gpu 0 --model VIT_base_32 --acc_cutoff 10

```

<h3>Positive and Negative Perturbation Tests (Table 3)</h3>

You must have ILSVRC2012_img_val.tar and point these tests to it.


```
python3 generate_visualizations.py --method IG  --imagenet-validation-path <path-to-imagenet-tar> --vis-class top --model VIT_base_32
python3 generate_visualizations.py --method GC  --imagenet-validation-path <path-to-imagenet-tar> --vis-class top --model VIT_base_32
python3 generate_visualizations.py --method LRP  --imagenet-validation-path <path-to-imagenet-tar> --vis-class top --model VIT_base_32
python3 generate_visualizations.py --method Transition_attn  --imagenet-validation-path <path-to-imagenet-tar> --vis-class top --model VIT_base_32
python3 generate_visualizations.py --method Bidirectional  --imagenet-validation-path <path-to-imagenet-tar> --vis-class top --model VIT_base_32
python3 generate_visualizations.py --method TIS  --imagenet-validation-path .<path-to-imagenet-tar> --vis-class top --model VIT_base_32
python3 generate_visualizations.py --method VIT_CX  --imagenet-validation-path <path-to-imagenet-tar> --vis-class top --model VIT_base_32
python3 generate_visualizations.py --method Calibrate_Ordered  --imagenet-validation-path <path-to-imagenet-tar> --vis-class top --model VIT_base_32

python3 pertubation_eval_from_hdf5.py --method IG --vis-class top --model VIT_base_32
python3 pertubation_eval_from_hdf5.py --method GC --vis-class top --model VIT_base_32
python3 pertubation_eval_from_hdf5.py --method LRP --vis-class top --model VIT_base_32
python3 pertubation_eval_from_hdf5.py --method Transition_attn --vis-class top --model VIT_base_32
python3 pertubation_eval_from_hdf5.py --method Bidirectional --vis-class top --model VIT_base_32
python3 pertubation_eval_from_hdf5.py --method TIS --vis-class top --model VIT_base_32
python3 pertubation_eval_from_hdf5.py --method VIT_CX --vis-class top --model VIT_base_32
python3 pertubation_eval_from_hdf5.py --method Calibrate_Ordered --vis-class top --model VIT_base_32

python3 pertubation_eval_from_hdf5.py --method IG --vis-class top --neg --model VIT_base_32
python3 pertubation_eval_from_hdf5.py --method GC --vis-class top --neg --model VIT_base_32
python3 pertubation_eval_from_hdf5.py --method LRP --vis-class top --neg --model VIT_base_32
python3 pertubation_eval_from_hdf5.py --method Transition_attn --vis-class top --neg --model VIT_base_32
python3 pertubation_eval_from_hdf5.py --method Bidirectional --vis-class top --neg --model VIT_base_32
python3 pertubation_eval_from_hdf5.py --method TIS --vis-class top --neg --model VIT_base_32
python3 pertubation_eval_from_hdf5.py --method VIT_CX --vis-class top --neg --model VIT_base_32
python3 pertubation_eval_from_hdf5.py --method Calibrate_Ordered --vis-class top --neg --model VIT_base_32

```

<h3>Monotonicity Tests (Table 6)</h3>

```
python3 MDATestMonotonicity.py --function IG --model VIT_base_32 --image_count 100 --imagenet <path-to-imagenet-data> --gpu 0
python3 MDATestMonotonicity.py --function GC --model VIT_base_32 --image_count 100 --imagenet <path-to-imagenet-data> --gpu 0
python3 MDATestMonotonicity.py --function LRP --model VIT_base_32 --image_count 100 --imagenet <path-to-imagenet-data> --gpu 0
python3 MDATestMonotonicity.py --function Transition_attn --model VIT_base_32 --image_count 100 --imagenet <path-to-imagenet-data> --gpu 0
python3 MDATestMonotonicity.py --function Bidirectional --model VIT_base_32 --image_count 100 --imagenet <path-to-imagenet-data> --gpu 0
python3 MDATestMonotonicity.py --function TIS --model VIT_base_32 --image_count 100 --imagenet <path-to-imagenet-data> --gpu 0
python3 MDATestMonotonicity.py --function VIT_CX --model VIT_base_32 --image_count 100 --imagenet <path-to-imagenet-data> --gpu 0
python3 MDATestMonotonicity.py --function Calibrate --model VIT_base_32 --image_count 100 --imagenet <path-to-imagenet-data> --gpu 0
python3 MDATestMonotonicity.py --function Calibrate_Ins --model VIT_base_32 --image_count 100 --imagenet <path-to-imagenet-data> --gpu 0
python3 MDATestMonotonicity.py --function Calibrate_Del --model VIT_base_32 --image_count 100 --imagenet <path-to-imagenet-data> --gpu 0
```

<h3>Tau Ablation Study (Figure 7, Tables 7, 8, 9)</h3>

```
python3 MDATest.py --function Calibrate_Cutoff --cutoff 0 --model VIT_base_32 --image_count 100 --imagenet <path-to-imagenet-data> --gpu 0
python3 MDATest.py --function Calibrate_Cutoff --cutoff 0.1 --model VIT_base_32 --image_count 100 --imagenet <path-to-imagenet-data> --gpu 0
python3 MDATest.py --function Calibrate_Cutoff --cutoff 0.2 --model VIT_base_32 --image_count 100 --imagenet <path-to-imagenet-data> --gpu 0
python3 MDATest.py --function Calibrate_Cutoff --cutoff 0.3 --model VIT_base_32 --image_count 100 --imagenet <path-to-imagenet-data> --gpu 0
python3 MDATest.py --function Calibrate_Cutoff --cutoff 0.4 --model VIT_base_32 --image_count 100 --imagenet <path-to-imagenet-data> --gpu 0
python3 MDATest.py --function Calibrate_Cutoff --cutoff 0.5 --model VIT_base_32 --image_count 100 --imagenet <path-to-imagenet-data> --gpu 0
python3 MDATest.py --function Calibrate_Cutoff --cutoff 0.6 --model VIT_base_32 --image_count 100 --imagenet <path-to-imagenet-data> --gpu 0
python3 MDATest.py --function Calibrate_Cutoff --cutoff 0.7 --model VIT_base_32 --image_count 100 --imagenet <path-to-imagenet-data> --gpu 0
python3 MDATest.py --function Calibrate_Cutoff --cutoff 0.8 --model VIT_base_32 --image_count 100 --imagenet <path-to-imagenet-data> --gpu 0
python3 MDATest.py --function Calibrate_Cutoff --cutoff 0.9 --model VIT_base_32 --image_count 100 --imagenet <path-to-imagenet-data> --gpu 0
python3 MDATest.py --function Calibrate_Cutoff --cutoff 1 --model VIT_base_32 --image_count 100 --imagenet <path-to-imagenet-data> --gpu 0
```

<h3>Kappa Ablation Study (Figure 9)</h3>

```
python3 evalKappaMDA.py --model VIT_base_16 --image_count 100 --imagenet <path-to-imagenet-data> --gpu 0
```


<h3>Runtime Tests (Table 10)</h3>

```
python3 timeComparison.py --function IG --model VIT_base_16 --image_count 100 --imagenet <path-to-imagenet-data> --gpu 0
python3 timeComparison.py --function GC --model VIT_base_16 --image_count 100 --imagenet <path-to-imagenet-data> --gpu 0
python3 timeComparison.py --function LRP --model VIT_base_16 --image_count 100 --imagenet <path-to-imagenet-data> --gpu 0
python3 timeComparison.py --function Transition_attn --model VIT_base_16 --image_count 100 --imagenet <path-to-imagenet-data> --gpu 0
python3 timeComparison.py --function Bidirectional --model VIT_base_16 --image_count 100 --imagenet <path-to-imagenet-data> --gpu 0
python3 timeComparison.py --function TIS --model VIT_base_16 --image_count 100 --imagenet <path-to-imagenet-data> --gpu 0
python3 timeComparison.py --function VIT_CX --model VIT_base_16 --image_count 100 --imagenet <path-to-imagenet-data> --gpu 0
python3 timeComparison.py --function Calibrate --model VIT_base_16 --image_count 100 --imagenet <path-to-imagenet-data> --gpu 0

python3 timeComparison.py --function IG --model VIT_base_32 --image_count 100 --imagenet <path-to-imagenet-data> --gpu 0
python3 timeComparison.py --function GC --model VIT_base_32 --image_count 100 --imagenet <path-to-imagenet-data> --gpu 0
python3 timeComparison.py --function LRP --model VIT_base_32 --image_count 100 --imagenet <path-to-imagenet-data> --gpu 0
python3 timeComparison.py --function Transition_attn --model VIT_base_32 --image_count 100 --imagenet <path-to-imagenet-data> --gpu 0
python3 timeComparison.py --function Bidirectional --model VIT_base_32 --image_count 100 --imagenet <path-to-imagenet-data> --gpu 0
python3 timeComparison.py --function TIS --model VIT_base_32 --image_count 100 --imagenet <path-to-imagenet-data> --gpu 0
python3 timeComparison.py --function VIT_CX --model VIT_base_32 --image_count 100 --imagenet <path-to-imagenet-data> --gpu 0
python3 timeComparison.py --function SHAP --model VIT_base_32 --image_count 100 --imagenet <path-to-imagenet-data> --gpu 0
python3 timeComparison.py --function Calibrate --model VIT_base_32 --image_count 100 --imagenet <path-to-imagenet-data> --gpu 0

python3 timeComparison.py --function IG --model VIT_tiny_16 --image_count 100 --imagenet <path-to-imagenet-data> --gpu 0
python3 timeComparison.py --function GC --model VIT_tiny_16 --image_count 100 --imagenet <path-to-imagenet-data> --gpu 0
python3 timeComparison.py --function LRP --model VIT_tiny_16 --image_count 100 --imagenet <path-to-imagenet-data> --gpu 0
python3 timeComparison.py --function Transition_attn --model VIT_tiny_16 --image_count 100 --imagenet <path-to-imagenet-data> --gpu 0
python3 timeComparison.py --function Bidirectional --model VIT_tiny_16 --image_count 100 --imagenet <path-to-imagenet-data> --gpu 0
python3 timeComparison.py --function TIS --model VIT_tiny_16 --image_count 100 --imagenet <path-to-imagenet-data> --gpu 0
python3 timeComparison.py --function VIT_CX --model VIT_tiny_16 --image_count 100 --imagenet <path-to-imagenet-data> --gpu 0
python3 timeComparison.py --function Calibrate --model VIT_tiny_16 --image_count 100 --imagenet <path-to-imagenet-data> --gpu 0
```

<h3>Qualitative results (End of Appendix)</h3>

```
python3 makeMDApdf.py --model VIT_base_16  --imagenet <path-to-imagenet-data> --gpu 0
python3 makeMDApdf.py --model VIT_tiny_16  --imagenet <path-to-imagenet-data> --gpu 0
python3 makeMDApdf.py --model VIT_base_32  --imagenet <path-to-imagenet-data> --gpu 0
```
