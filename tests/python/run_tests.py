import os
import sys
import argparse
import unittest


class DLTestConfig:
    def __init__(self):
        self.modules_path = ""


def parse_options(config: DLTestConfig):
    usage = "Usage: python {} [-p <modules_path>]".format(__file__)
    parser = argparse.ArgumentParser(usage)
    parser.add_argument("-p", dest="modules_path", type=str, help="fake-bpy-module path")

    args = parser.parse_args()
    if args.modules_path:
        config.modules_path = args.modules_path


def main():
    config = DLTestConfig()
    parse_options(config)

    path = os.path.abspath(config.modules_path)
    sys.path.append(path)

    sys.path.append(os.path.dirname(__file__))
    import dl_test

    test_cases = [
        dl_test.layers.element_wise_test.AddLayerTest,
        dl_test.layers.element_wise_test.MulLayerTest,
        dl_test.layers.linear_test.LinearLayerTest,
        dl_test.layers.activation_test.TanhLayerTest,
        dl_test.layers.activation_test.ReluLayerTest,
        dl_test.layers.activation_test.SigmoidLayerTest,
        dl_test.layers.activation_test.SoftmaxLayerTest,
        dl_test.layers.convolution_test.Convolution2DLayerTest,
        dl_test.layers.normalization_test.BatchNormalizationLayerTest,
        dl_test.layers.dropout_test.DropoutLayerTest,
    ]

    suite = unittest.TestSuite()
    for case in test_cases:
        suite.addTest(unittest.makeSuite(case))
    ret = unittest.TextTestRunner().run(suite).wasSuccessful()
    sys.exit(not ret)


if __name__ == "__main__":
    main()
