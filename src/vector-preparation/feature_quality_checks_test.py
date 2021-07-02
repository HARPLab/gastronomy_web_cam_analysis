import json
import unittest
import numpy as np
import cv2

import sys
sys.path.append("..")
import qchecks
import arconsts

import feature_quality_checks

class TestMirrorMethods(unittest.TestCase):

	# Verify probabilities remain intact
	# Verify missing entries remain zeros
	# verify all are in bounds of image (no negatives in particular)
	# verify that keypoints transform roughly correctly
	# Run on a set pose

	def test_homography(self):
		h1 = feature_quality_checks.get_homography_A_to_B()
		h2 = feature_quality_checks.get_homography_B_to_A()

		# print(h1)
		# print(h2)

		together = np.dot(h1, h2)
		# print(together)

		are_equal = np.array_equal(h1, h2)

		self.assertEqual(are_equal, False)


	def test_mirror_transformation(self):
		set_a = arconsts.seat_keypoints_A
		set_b = arconsts.seat_keypoints_B
		num_points = set_a.shape[0]

		h1 = feature_quality_checks.get_homography_A_to_B()
		h2 = feature_quality_checks.get_homography_B_to_A()

		print("h a to b")
		print(h1)
		print("h b to a")
		print(h2)

		flipped_a = feature_quality_checks.get_mirror_pose(set_b, h1)
		flipped_b = feature_quality_checks.get_mirror_pose(set_b, h2)

		# print(set_a)
		print("OG b")
		print(set_b)
		print("b times a to b")
		print(flipped_a)
		print("b times b to a")
		print(flipped_b)
		print("OG a")
		print(set_a)

	# Verify not-found zero points remain the same post-transform
	def test_zer0_entries(self):
		set_a = arconsts.seat_keypoints_A
		set_a[1][0] = 0
		set_a[1][1] = 0

		h1 = feature_quality_checks.get_homography_A_to_B()
		flipped_a = feature_quality_checks.get_mirror_pose(set_a, h1)

		x1 = flipped_a[1]
		x2 = set_a[1]

		self.assertEqual(x1[0], x2[0])
		self.assertEqual(x1[1], x2[1])

if __name__ == '__main__':
	np.set_printoptions(suppress=True)
	unittest.main()




