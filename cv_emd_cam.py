import argparse
from paz.backend.camera import VideoPlayer
from paz.backend.camera import Camera
from paz.pipelines import DetectMiniXceptionFER


def emdcam(camera_id=0, offset=0.1):
	pipeline = DetectMiniXceptionFER([offset, offset])
	camera = Camera(camera_id)
	player = VideoPlayer((640, 480), pipeline, camera)
	player.run()
	player.stop()

#emdcam()
""" import argparse
from paz.backend.camera import VideoPlayer
from paz.backend.camera import Camera
from paz.pipelines import DetectMiniXceptionFER


if __name__ == "__main__":
	parser = argparse.ArgumentParser(description='Real-time face classifier')
	parser.add_argument('-c', '--camera_id', type=int, default=0,
									help='Camera device ID')
	parser.add_argument('-o', '--offset', type=float, default=0.1,
									help='Scaled offset to be added to bounding boxes')
	args = parser.parse_args()

	pipeline = DetectMiniXceptionFER([args.offset, args.offset])
	camera = Camera(args.camera_id)
	player = VideoPlayer((640, 480), pipeline, camera)
	player.run()
	player.stop()
					 """

