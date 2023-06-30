import sys

sys.path.append("../")
from build.pybind_interface.ReID import ReID
import cv2
import time
import numpy as np

import argparse



if __name__ == '__main__':

    arg_parser = argparse.ArgumentParser(description="ReID")
    arg_parser.add_argument("--serialize-name", type=str, default="../build/sbs_R50-ibn.engine", help="model serialize name")
    arg_parser.add_argument("--image-path", type=str, default="../data/Market-1501-v15.09.15/calib_set/-1_c1s2_009916_03.jpg", help="image path")
    arg_parser.add_argument("--iter-nums", type=int, default=10, help="iteration numbers")

    args = arg_parser.parse_args()
    iter_ = args.iter_nums
    # REID(GPU_ID)
    m = ReID(0)
    m.build(args.serialize_name)
    print("build done")
    
    frame = cv2.imread(args.image_path)



    feat = m.infer(frame)

    # list[] 2048
    
    # save feature
    np.save("./feat.npy", feat)


    # for batch infer
    # batch_frames = []
    # for i in range(iter_):
    #     batch_frames.append(frame)
    #     # m.infer(frame)

    # t0 = time.time()
    # m.batch_infer(batch_frames)
    # total = time.time() - t0
    # # print("CPP API fps is {:.1f}, avg infer time is {:.2f}ms".format(iter_ / total, total / iter_ * 1000))

    # print(f"batch size:{iter_}, total time:{total * 1000}ms")