# Association evaluation: pose error

This script evaluates the pose error metrics (median angular rotation error, median angular translation error, median absolute translation error, and pose error auc) in the line association problem.
The result of the script is a `.json` file containing the specified metrics.

## Script running
It is necessary to provide folders containing lines, associations between frame pairs, and depth maps to calculate metrics. The file with associations must contain records of the form `i,j`,
where `i` corresponds to the index in the line array of the first frame,
and `j` corresponds to the index in the line array of the second frame.
Also, a file containing poses for the corresponding frame depth maps in [TUM format](https://cvg.cit.tum.de/data/datasets/rgbd-dataset/file_formats) is needed.
In addition, it is necessary to provide a file containing a calibration matrix
and a file with associations between depth frames and images in which lines were detected.
Examples can be found in `example` folder.

```bash
usage: python evaluate.py [-h] --lines PATH --associations PATH --depths PATH
                          --poses PATH --depth-associations PATH
                          --calibration-matrix PATH
                          [--pose-error-auc-thresholds NUM [NUM ...]] --output
                          PATH [--output-file STR]

arguments:
  -h, --help            show this help message and exit
  --lines PATH, -l PATH
                        path to the folder with lines (default: None)
  --associations PATH, -a PATH
                        path to the folder with line associations (default:
                        None)
  --depths PATH, -d PATH
                        path to the folder with depth maps (default: None)
  --poses PATH, -p PATH
                        path to the file with poses (default: None)
  --depth-associations PATH, -A PATH
                        path to the file with associations between images and
                        depth maps (default: None)
  --calibration-matrix PATH, -c PATH
                        path to the file with calibration matrix (default:
                        None)
  --pose-error-auc-thresholds NUM [NUM ...], -t NUM [NUM ...]
                        thresholds in degrees for angular error auc
                        calculation (default: [1.0, 3.0, 5.0, 10.0])
  --output PATH, -o PATH
                        output path (default: None)
  --output-file STR, -O STR
                        name of output file (default: pose_errors.json)
```

### Example
```bash
python evaluate.py \
-l ./example/lines \
-a ./example/associations \
-d ./example/depth \
-p ./example/poses.txt \
-A ./example/depth_associations.txt \
-c ./example/calibration_matrix.txt \
-t 1 3 \
-o ./output
```
