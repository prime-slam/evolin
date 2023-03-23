# Copyright (c) 2023, Kirill Ivanov, Anastasiia Kornilova and Dmitrii Iarosh
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

import g2o
import numpy as np

from src.typing import Array4x4


class RelativePoseEstimator:
    def __init__(self):
        self.iterations_number = 50
        self.optimizer_iterations_number = 30
        self.reprojection_threshold = 1000
        self.edges_min_number = 12

    def estimate(
        self,
        first_lines,
        second_lines,
        second_lines_3d,
        calibration_matrix,
        first_to_second=True,
    ) -> Array4x4[float]:
        optimizer = self.__create_optimizer()
        v1 = g2o.VertexSE3Expmap()
        v1.set_id(0)
        v1.set_fixed(False)
        optimizer.add_vertex(v1)
        line_edges = []
        v1.set_estimate(g2o.SE3Quat(np.eye(3), np.zeros((3,))))

        lines_number = len(second_lines)
        information = np.eye(3)

        for i in range(lines_number):
            measurement = np.cross(
                np.append(first_lines[i][0], 1), np.append(first_lines[i][1], 1)
            )
            first_edge = self.__create_edge(
                v1, measurement, information, calibration_matrix, second_lines_3d[i][0]
            )
            second_edge = self.__create_edge(
                v1, measurement, information, calibration_matrix, second_lines_3d[i][1]
            )
            optimizer.add_edge(first_edge)
            optimizer.add_edge(second_edge)
            line_edges.append((first_edge, second_edge))

        inlier_mask = np.ones(lines_number, dtype=bool)

        for i in range(self.iterations_number):
            v1.set_estimate(g2o.SE3Quat(np.eye(3), np.zeros((3,))))
            optimizer.initialize_optimization()
            optimizer.optimize(self.optimizer_iterations_number)

            for j, (first_edge, second_edge) in enumerate(line_edges):
                if (
                    first_edge.chi2() + second_edge.chi2()
                ) / 2 > self.reprojection_threshold / (i + 1) ** 2:
                    inlier_mask[j] = False
                    first_edge.set_level(1)
                    second_edge.set_level(1)
                else:
                    inlier_mask[j] = True
                    first_edge.set_level(0)
                    second_edge.set_level(0)

                if i == self.iterations_number - 2:
                    first_edge.set_robust_kernel(None)
                    second_edge.set_robust_kernel(None)

            if 2 * np.count_nonzero(inlier_mask) < self.edges_min_number:
                break

        second_to_first_pose = v1.estimate().matrix()
        return (
            np.linalg.inv(second_to_first_pose)
            if first_to_second
            else second_to_first_pose
        )

    @staticmethod
    def __create_optimizer():
        optimizer = g2o.SparseOptimizer()
        block_solver = g2o.BlockSolverSE3(g2o.LinearSolverEigenSE3())
        solver = g2o.OptimizationAlgorithmLevenberg(block_solver)
        optimizer.set_algorithm(solver)

        return optimizer

    @staticmethod
    def __create_edge(v1, measurement, information, calibration_matrix, Xw):
        fx = calibration_matrix[0, 0]
        fy = calibration_matrix[1, 1]
        cx = calibration_matrix[0, 2]
        cy = calibration_matrix[1, 2]

        edge = g2o.EdgeLineProjectXYZOnlyPose()
        edge.set_vertex(0, v1)
        edge.set_measurement(measurement)
        edge.set_information(information)
        edge.set_robust_kernel(g2o.RobustKernelHuber())
        edge.fx = fx
        edge.fy = fy
        edge.cx = cx
        edge.cy = cy
        edge.Xw = Xw

        return edge
