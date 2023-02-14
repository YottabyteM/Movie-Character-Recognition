from Pose_detect import Pose_detect
import argparse
protoFile = "./model/coco/pose_deploy_linevec.prototxt"
weightsFile = "./model/coco/pose_iter_440000.caffemodel"
VedioPath = "./videos/1.mp4"
parser = argparse.ArgumentParser(description='Run keypoint detection')
parser.add_argument("--device", default="gpu", help="Device to inference on")
args = parser.parse_args()
if __name__ == '__main__':
    model = Pose_detect(protoFile, weightsFile, args)
    model.predict(VedioPath)