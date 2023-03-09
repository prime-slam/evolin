# Detection evaluation: vectorized classification

This script evaluates the metrics in the line detection problem based on vector line representation.
Orthogonal and structural distances establish true positive lines within their specified maximum values.
In the case of lines with scores, average precision and PR curves will be calculated.
If a file with thresholds for line scores were passed as an argument, precision, recall, and F-score would also be built on them.
In the case of lines without scores, only precision, recall, and F-score will be calculated.
The result of the script is a `.json` file containing the specified metrics.

## Script running
To calculate metrics, you need folders containing ground truth and predicted lines, images, scores, if available, and their thresholds.
Examples can be found in `example` folder.

```bash
usage: python evaluate.py [-h] --pred-lines PATH [--scores PATH]
                          [--score-thresholds PATH] --output PATH --imgs PATH
                          --gt-lines PATH
                          [--distance-thresholds SEQ [SEQ ...]]
                          [--output-file STR]

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
                        output path (default: None)
  --imgs PATH, -i PATH  path to images (default: None)

  --distance-thresholds SEQ [SEQ ...], -d SEQ [SEQ ...]
                        distance thresholds in pixels (default: [5.0, 10.0,
                        15.0])
  --output PATH, -o PATH
  --output-file STR, -O STR
                        name of output file (default:
                        vectorized_classification_metrics.json)
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
