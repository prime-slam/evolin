# Copyright (c) 2022, Kirill Ivanov, Anastasiia Kornilova and Dmitrii Iarosh
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse


def create_base_parser(prog: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=prog,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--pred-lines",
        "-p",
        metavar="PATH",
        help="path to the folder with predicted lines",
        required=True,
    )

    parser.add_argument(
        "--scores",
        "-s",
        metavar="PATH",
        help="path to the folder with line scores",
        default=None,
    )

    parser.add_argument(
        "--score-thresholds",
        "-S",
        metavar="PATH",
        help="path to the file with score thresholds",
        default=None,
    )

    parser.add_argument(
        "--output",
        "-o",
        metavar="PATH",
        help="output path",
        required=True,
    )

    return parser
