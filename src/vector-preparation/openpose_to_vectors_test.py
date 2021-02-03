import openpose_to_vectors
import json
import unittest
import numpy as np

class TestRoleAssignmentMethods(unittest.TestCase):

	def test_upper(self):
		test_in = [(133, 193), (302, 257), (385, 165), (140, 329)]
		correct_A = [True, False, False, True]
		correct_B = [False, True, False, False]

		bd_box_A = openpose_to_vectors.bd_box_A
		bd_box_B = openpose_to_vectors.bd_box_B

		for i in range(len(test_in)):
			pt = test_in[i]
			self.assertEqual(openpose_to_vectors.in_bd_box(bd_box_A, pt), correct_A[i])
			self.assertEqual(openpose_to_vectors.in_bd_box(bd_box_B, pt), correct_B[i])



		

	def test_which_pose(self):
		filename = '8-21-18_shows_quality-check-random_f104051.json'
		filename = 'quality-checks/' + filename

		with open(filename) as f:
			data = json.load(f)

	
		pose_a = data['pose-A']
		pose_b = data['pose-B']

		all_poses = data['all-poses']

		all_poses_reshaped = []
		for pose in all_poses:
			all_poses_reshaped.append(np.asarray(pose).reshape((25, 3)))


		test_a, test_b = openpose_to_vectors.get_role_assignments(all_poses_reshaped)

		# print("Test A")
		# print(test_a)

		# print("Test B")
		# print(test_b)
		test_a = test_a.aslist()
		test_b = test_b.aslist()

		self.assertListEqual(pose_a, test_a)
		self.assertListEqual(pose_b, test_b)

if __name__ == '__main__':
	np.set_printoptions(suppress=True)
	unittest.main()




