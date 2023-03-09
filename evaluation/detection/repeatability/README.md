# Detection evaluation: repeatability

This script evaluates the repeatability metrics in the line detection problem.
Orthogonal and structural distances establish true positive lines within their specified maximum values.
In the case of lines with scores, it is necessary to pass the score thresholds by which the lines will be filtered.
Otherwise, the metrics will be calculated for all lines.
The result of the script is a `.json` file containing the repeatability score and localization errors.

## Script running
It is necessary to provide folders containing predicted lines and depth maps to calculate metrics.
Also, a file containing poses for the corresponding frame depth maps in TUM format is needed.
In addition, it is necessary to provide a file containing a calibration matrix
and a file with associations between depth frames and images in which lines were detected.
Examples can be found in `example` folder.

```bash
usage: python evaluate.py [-h] --pred-lines PATH [--scores PATH]
                          [--score-thresholds PATH] --output PATH --depths
                          PATH --poses PATH --associations PATH
                          --calibration-matrix PATH
                          [--distance-thresholds SEQ [SEQ ...]]
                          [--frames-steps SEQ [SEQ ...]] [--output-file STR]

arguments:
  -h, --help            show this help message and exit
  --pred-lines PATH, -p PATH
                        path to the folder with predicted lines (default:
                        None)
  --scores PATH, -s PATH
                        path to the folder with line scores (default: None)
  --score-thresholds PATH, -S PATH
                        path to the file with score thresholds (default: None)
  --depths PATH, -D PATH
                        path to the folder with depth maps (default: None)
  --poses PATH, -P PATH
                        path to the file with poses (default: None)
  --depth-associations PATH, -a PATH
                        path to the file with associations between images and
                        depth maps (default: None)
  --calibration-matrix PATH, -c PATH
                        path to the file with calibration matrix (default:
                        None)
  --distance-thresholds SEQ [SEQ ...], -d SEQ [SEQ ...]
                        distance thresholds in pixels (default: [5.0, 10.0,
                        15.0])
  --frames-steps SEQ [SEQ ...], -f SEQ [SEQ ...]
                        distance thresholds in pixels (default: [10])
  --output PATH, -o PATH
                        output path (default: None)
  --output-file STR, -O STR
                        name of output file (default:
                        repeatability_metrics.json)
```

### Examples
**Lines with scores**
```bash
python evaluate.py
-p
./example/scored_lines/lines
-s
./example/scored_lines/scores
-S
./example/scored_lines/score_thresholds.txt
-D
./example/depth
-P
./example/poses.txt
-a
./example/depth_associations.txt
-c
./example/calibration_matrix.txt
-d 5 10 15
-f 1
-o
./output
```
**Lines without scores**
```bash
python evaluate.py
-p
./example/unscored_lines
-D
./example/depth
-P
./example/poses.txt
-a
./example/depth_associations.txt
-c
./example/calibration_matrix.txt
-d 5 10 15
-f 1
-o
./output
```
