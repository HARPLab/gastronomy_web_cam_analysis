from feature_utils import *

class RestaurantFrame:
        poses_arrays_raw = []
        num_poses_raw = 0
        pa_label = 'NONE'
        pb_label = 'NONE'
        frame_number = -1

        poses_arrays_cleaned = []
        num_poses_cleaned = 0

        pose_PA = None
        pose_PB = None
        # waiter is always just ONE of the waiters
        pose_waiter = None

        delta_PA = None
        delta_PB = None
        prev = None

        max_detected = 9

        def set_PA(self, pose):
                #TODO handle None case nicely
                self.pose_PA = pose

        def set_PB(self, pose):
                self.pose_PB = pose

        def set_waiter(self, pose):
                self.pose_waiter = pose

        def get_PA(self):
                return self.pose_PA

        def get_PB(self):
                return self.pose_PB

        def get_waiter(self):
                return self.pose_waiter

        def set_previous_processed_frame(self, frame):
                self.delta_PA = frame.get_PA() - self.get_PA()
                self.delta_PB = frame.get_PB() - self.get_PB()

                self.prev = frame

        def get_delta_PA(self):
                return self.delta_PA

        def get_delta_PB(self):
                return self.delta_PB

        def get_prev(self):
                return self.prev

        def set_roles(self, pa, pb, w):
                self.set_PA(pa)
                self.set_PB(pb)
                self.set_waiter(w)

        def set_PA(self, pa):
                self.pose_PA = pa

        def set_PB(self, pb):
                self.pose_PB = pb
        def get_sift(self):
                features = self.sift_features.split("),")
                features[0] = features[0][1:]
                features[len(features)-1] = features[len(features)-1][0:-2]
                features = [x[2:] for x in features]
                sift_feat = np.zeros((len(features), 2))
                for i,f in enumerate(features):
                    x = f.split(", ")[0]
                    y = f.split(", ")[1]
                    if len(x) < 5 and len(y) < 5 and len(x) > 0 and len(y)>0:
                        sift_feat[i][0] = int(x)
                        sift_feat[i][1] = int(y)
                return list(sift_feat.flatten())
        def set_waiter(self, w):
                self.pose_waiter = w
        def __init__(self, frame_num, frame, sift):
                self.sift_features = sift
                self.frame_number = frame_num

                # 25 keyframes for body
                start = frame.find(":") + len(":")
                end = frame.find("LH:")
                poses = frame[start:end]
                # print(poses)
                # Overall pose info
                # list of people, with 25 3d points per person

                poses = poses[1:-1]
                pose_list = poses.split("[[")
                pose_list = pose_list[1:]

                processed_poses = []

                num_people = len(pose_list)
                num_points_total = 0

                for pose in pose_list:
                        pose = pose.replace("]]", "]")
                        pose = pose.rstrip(']')
                        pose = pose.lstrip('[')


                        pose_points = pose.split(']\n  ')
                        pt_set = []
                        for pt in pose_points:
                                pt = pt.lstrip('[')
                                pt = pt.replace('\n', '')
                                pt = pt.replace(']', '')

                                p = pt.split(" ")
                                p = list(filter(None, p))
                                (x,y,z) = p

                                pt_set.append((float(x), float(y), float(z)))

                        processed_poses.append([pt_set])
                #print(processed_poses)
                self.poses_arrays_raw = processed_poses
                self.num_poses_raw = num_people

                start = frame.find("LH:") + len("LH:")
                end = frame.find("RH:")
                lh = frame[start:end]
                # print(lh)
                # frame_obj['lh'] = float(lh)

                start = frame.find("RH:") + len("RH:")
                end = frame.find("Face:")
                rh = frame[start:end]
                # frame_obj['rh'] = float(rh)

                start = frame.find("Face:") + len("Face:")
                end = frame.find("PA:")
                face = frame[start:end]
                # frame_obj['face'] = float(face)

                start = frame.find("PA:") + len("PA:")
                end = frame.find(" PB:")
                pa = frame[start:end]
                self.pa_label = pa

                start = frame.find("PB:") + len("PB:")
                pb = frame[start:].strip()
                self.pb_label = pb

        def get_poses_raw(self):
                return self.poses_arrays_raw

        def get_poses_clean(self):
                padded_array = []
                for i in range(self.max_detected):
                        if len(self.poses_arrays_cleaned) > i:
                                padded_array.append(self.poses_arrays_cleaned[i])
                        else:
                                padded_array.append(NULL_POSE)

                return padded_array

        def get_label_PA(self):
                return self.pa_label

        def get_label_PB(self):
                return self.pb_label

        def get_num_poses_raw(self):
                return self.num_poses_raw

        def get_num_poses_clean(self):
                return self.num_poses_cleaned

        def get_frame_number(self):
                return self.frame_number

