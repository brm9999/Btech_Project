import sys
if sys.version_info[0] < 3 and sys.version_info[1] < 2:
	raise Exception("Must be using >= Python 3.2")

from os import listdir, path

if not path.isfile('face_detection/detection/sfd/s3fd.pth'):
	raise FileNotFoundError('Save the s3fd model to face_detection/detection/sfd/s3fd.pth \
							before running this script!')

import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
import argparse, os, cv2, traceback, subprocess
from tqdm import tqdm
from glob import glob
from synthesizer import audio
from synthesizer.hparams import hparams as hp

import face_detection

parser = argparse.ArgumentParser()

parser.add_argument('--ngpu', help='Number of GPUs across which to run in parallel', default=1, type=int)
parser.add_argument('--batch_size', help='Single GPU Face detection batch size', default=2, type=int)
parser.add_argument("--resize_factor", help="Resize the frames before face detection", default=1, type=int)


args = parser.parse_args()

fa = [face_detection.FaceAlignment(face_detection.LandmarksType._2D, flip_input=False, 
									device='cuda:{}'.format(id)) for id in range(1)]

template = 'ffmpeg -loglevel panic -y -i {} -ar {} -f wav {}'
template2 = 'ffmpeg -hide_banner -loglevel panic -threads 1 -y -i {} -async 1 -ac 1 -vn -acodec pcm_s16le -ar 16000 {}'


def process_video_file(args, gpu_id):
	video_stream = cv2.VideoCapture(0)
	count = 0
	fulldir = 'C:/Project/realtime/checking'
	storage= 'C:/Project/realtime/finaldata'
	os.makedirs(fulldir, exist_ok=True)
	os.makedirs(storage, exist_ok=True)
	i =-1

	while 1:
		
		still_reading, frame = video_stream.read()
		if not still_reading:
			video_stream.release()
			break
		cv2.imwrite(os.path.join(fulldir, "frame{:d}.jpg".format(count)), frame)
		count+=1
		frame = cv2.resize(frame, (frame.shape[1]//args.resize_factor, frame.shape[0]//args.resize_factor))
		
		preds = fa[0].get_detections_for_batch(np.asarray([frame]))

		for j, f in enumerate(preds):
			i += 1
			if f is None:
				continue

			cv2.imwrite(os.path.join(storage, '{}.jpg'.format(i)), f[0])
		
		
def main(args):
    
	print('Started processing with {} GPUs'.format(1))
	try:
		process_video_file(args, 0)
		
	except KeyboardInterrupt:
		exit(0)
	except:
		traceback.print_exc()

if __name__ == '__main__':
	main(args)
