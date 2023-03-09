# Detection evaluation: heatmap classification

This script evaluates the heatmap metrics in the line detection problem.
The result of the script is a `.json` file containing heatmap average precision,
heatmap maximum F-score, and the PR curve in the case of lines with scores,
and if there are no scores, precision, recall, and F-score.

## Script running
To calculate metrics, you need folders containing ground truth and predicted lines, images, scores, if available, and their thresholds.
Examples can be found in `example` folder.

```bash
usage: python evaluate.py [-h] --pred-lines PATH [--scores PATH]
                          [--score-thresholds PATH] --output PATH --imgs PATH
                          --gt-lines PATH [--output-file STR]

arguments:
  -h, --help            show this help message and exit
  --pred-lines PATH, -p PATH
                        path to the folder with predicted lines (default:
                        None)
  --gt-lines PATH, -g PATH
                        path to the folder with ground truth lines (default:
                        None)
  --scores PATH, -s PATH
                        path to the folder with line scores (default: None)
  --score-thresholds PATH, -S PATH
                        path to the file with score thresholds (default: None)
  --imgs PATH, -i PATH  path to images (default: None)
  --output PATH, -o PATH
                        output path (default: None)
  --output-file STR, -O STR
                        name of output file (default: heatmap_metrics.json)
```

### Examples
**Lines with scores**
```bash
python evaluate.py
-p
./example/scored_lines/lines
-g
./example/gt_lines
-s
./example/scored_lines/scores
-S
./example/scored_lines/score_thresholds.txt
-i
./example/images
-o
./output
```
**Lines without scores**
```bash
python evaluate.py
-p
./example/unscored_lines
-g
./example/gt_lines
-i
./example/images
-o
./output
```
