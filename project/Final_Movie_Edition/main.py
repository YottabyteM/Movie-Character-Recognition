from main_model import main_model
import argparse

Pose_protoFile = "./model/coco/pose_deploy_linevec.prototxt"
Pose_weightsFile = "./model/coco/pose_iter_440000.caffemodel"
Fer_model_path = "model/Expression/FER_model.h5"
ActorPath = ".\photo"
VedioPath = "./videos/3.mp4"
parser = argparse.ArgumentParser(description='Run keypoint detection')
parser.add_argument("--device", default="gpu", help="Device to inference on")
args = parser.parse_args()

if __name__ == '__main__':
    name = [""]
    main_model = main_model(name, Pose_protoFile, Pose_weightsFile, args, ActorPath)
    main_model.predict(VedioPath)
