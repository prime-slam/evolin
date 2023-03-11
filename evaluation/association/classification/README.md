# Association evaluation: classification

This script evaluates the classification metrics (precision, recall, and F-score) in the line association problem.
The result of the script is a `.json` file containing the specified metrics.

## Script running
To calculate the metrics, you must create two folders containing ground truth and predicted associations.
Each folder must contain the same number of files with associations for pairs of frames.
The file with associations must contain records of the form `i,j`,
where `i` corresponds to the index in the line array of the first frame,
and `j` corresponds to the index in the line array of the second frame.
Examples can be found in `example` folder.

```bash
usage: python evaluate.py [-h] --pred-associations PATH --gt-associations PATH
                          --output PATH [--output-file STR]

arguments:
  -h, --help            show this help message and exit
  --pred-associations PATH, -p PATH
                        path to the folder with predicted associations
                        (default: None)
  --gt-associations PATH, -g PATH
                        path to the folder with ground truth associations
                        (default: None)
  --output PATH, -o PATH
                        output path (default: None)
  --output-file STR, -O STR
                        name of output file (default:
                        classification_metrics.json)
```

### Example
```bash
python evaluate.py \
-p ./example/pred_associations \
-g ./example/gt_associations \
-o ./output
```
