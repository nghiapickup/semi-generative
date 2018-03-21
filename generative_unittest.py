# Unittest
# supervised and semi-supervised
#
# @nghia n h | Yamada-lab

import numpy as np
import unittest
import copy
import MMM.NBText as nb


#
# GMMTest
#
class GMMTest(unittest.TestCase):

    def test_demo(self):
        self.assertEqual(1, 1)


#
# MMMTest
#
class AgglomerativeTreeTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.vector_test1 = nb.hierarchy_tree(sum_vector=np.full((1, 5),1)[0], element_id_list=[1])
        cls.vector_test2 = nb.hierarchy_tree(sum_vector=np.arange(5).reshape(1, 5)[0], element_id_list=[1])
        cls.data_group = [
            nb.hierarchy_tree(sum_vector=np.asarray([1,2,3,4]), element_id_list=[0], splitter_list=[]),
            nb.hierarchy_tree(sum_vector=np.asarray([1,2,3,4]), element_id_list=[1], splitter_list=[]),
            nb.hierarchy_tree(sum_vector=np.asarray([2,8,6,4]), element_id_list=[2], splitter_list=[]),
            nb.hierarchy_tree(sum_vector=np.asarray([2,8,6,4]), element_id_list=[3], splitter_list=[]),
            nb.hierarchy_tree(sum_vector=np.asarray([4,5,3,4]), element_id_list=[4], splitter_list=[]),
            nb.hierarchy_tree(sum_vector=np.asarray([4,5,3,4]), element_id_list=[5], splitter_list=[]),
            nb.hierarchy_tree(sum_vector=np.asarray([7,1,2,6]), element_id_list=[6], splitter_list=[]),
            nb.hierarchy_tree(sum_vector=np.asarray([7,1,2,6]), element_id_list=[7], splitter_list=[]),
            nb.hierarchy_tree(sum_vector=np.asarray([4,2,7,4]), element_id_list=[8], splitter_list=[]),
            nb.hierarchy_tree(sum_vector=np.asarray([4,2,7,4]), element_id_list=[9], splitter_list=[])
        ]
        cls.empty_data = nb.SslDataset()

    def test_distance_function(self):
        """
        vector1 = [1, 1, 1, 1, 1]
        vector2 = [0, 1, 2, 3, 4]

        * bin-to-bin:
        |vector1 - vector2| = [1, 0, 1, 2, 3]
        |vector1 + vector2| = [1, 2, 3, 4, 5]
        distance = 1/2 + 0/4 + 1/6 + 4/8 + 9/10 = 1.9 + 1/6

        * match:
        mass vector1 = [1, 2, 3, 4, 5]
        mass vector2 = [0, 1, 3, 6, 10]
        distance = 1 + 1 + 0 + 2 + 5 = 9

        """
        bin_bin = nb.AgglomerativeTree.bin_bin_distance(self.vector_test1, self.vector_test2)
        match = nb.AgglomerativeTree.match_distance(self.vector_test1, self.vector_test2)
        self.assertEqual(1.9 + 1/6, bin_bin, 'bin_to_bin distance mismatch')
        self.assertEqual(9, match, 'match distance mismatch')

    def test_build_hierarchy_tree(self):
        """
        data_group =[
                    [1,2,3,4] [1,2,3,4]
                    [2,8,6,4] [2,8,6,4]
                    [4,5,3,4] [4,5,3,4]
                    [7,1,2,6] [7,1,2,6]
                    [4,2,7,4] [4,2,7,4]]

        * bin_to_bin distance
        Agglomerative tree and merged distances
            Splitter id		  0   7   1   5	  2	  6   4	  8	  3
                              |   |   |	  |   |   |   |   |   |
                            0	1	2	3	4	5	8	9	6	7
                            |	|	|	|	|	|	|	|	|	|
                             \ / 	 \ /	 \ /	 \ /	 \ /
            Layer 0			  0		  0		  0		  0		  0
                              |		  |		  |		  |		  |
                              |		   \	 /		  |		  |
                              |			\   /		  |		  |
            Layer 1			  |		 1,(179487)	      |		  |
                              |			  |           |		  |
                              |			   \		 /		  |
                              |				\		/		  |
            Layer 2			  |			   1,534344172		  |
                              |				    |			  |
                               \			   /			  |
                                \			  /				  |
            Layer 3			      1,59772893				  |
                                       |       				  |
                                        \					 /
                                         \					/
                                          \	   			   /
            Layer 4			                  2,69241961

        * match distance
        Agglomerative tree and merged distances
            Splitter id		  0   7   1   5	  2	  6   4	  8	  3
                              |   |   |	  |   |   |   |   |   |
                            0	1	2	3	4	5	8	9	6	7
                            |	|	|	|	|	|	|	|	|	|
                             \ / 	 \ /	 \ /	 \ /	 \ /
            Layer 0			  0		  0		  0		  0		  0
                              |		  |		  |		  |		  |
                              |		  |		   \	 /		  |
                              |		  |			\   /		  |
            Layer 1			  |		  |	      	  5			  |
                              |		  |           |		      |
                              |		  |	   		   \		 /
                              |		  |    			\		/
            Layer 2			  |		  |		 		   6.5
                              |		  |		       		|
                              |		   \			   /
                              |		    \			  /
            Layer 3			  |			     13.(3)
                              |       		   |
                               \	     	  /
                                \			 /
                                 \	   	    /
            Layer 4			         22.5
        """
        # test bin-bin
        bin_to_bin_expected = nb.hierarchy_tree(sum_vector=np.asarray([36, 36, 42, 44]),
                                                element_id_list=[0, 1, 2, 3, 4, 5, 8, 9, 6, 7],
                                                splitter_list=[nb.splitter(id=0, order=0), nb.splitter(id=2, order=1),
                                                               nb.splitter(id=4, order=2), nb.splitter(id=3, order=5),
                                                               nb.splitter(id=6, order=4), nb.splitter(id=5, order=6),
                                                               nb.splitter(id=1, order=7), nb.splitter(id=8, order=3),
                                                               nb.splitter(id=7, order=8)])
        model = nb.AgglomerativeTree(self.empty_data, 'bin_bin_distance')
        bin_bin_data = copy.deepcopy(self.data_group)
        result = model.build_hierarchy_tree(bin_bin_data)
        self.assertTrue(np.array_equal(bin_to_bin_expected.sum_vector, result.sum_vector),
                        'Agglomerative tree bin_to_bin distance: sum_vector assertion fail')
        self.assertTrue(bin_to_bin_expected.element_id_list == result.element_id_list,
                        'Agglomerative tree bin_to_bin distance: element_id_list assertion fail')
        self.assertTrue((len(bin_to_bin_expected) == len(result)) and
                        all([x==y for x, y in zip(bin_to_bin_expected.splitter_list, result.splitter_list)]),
                        'Agglomerative tree bin_to_bin distance: splitter_list assertion fail')
        # test match
        march_expected = nb.hierarchy_tree(sum_vector=np.asarray([36, 36, 42, 44]),
                                           element_id_list=[0, 1, 2, 3, 4, 5, 8, 9, 6, 7],
                                           splitter_list=[nb.splitter(id=0, order=0), nb.splitter(id=2, order=1),
                                                          nb.splitter(id=4, order=2), nb.splitter(id=6, order=4),
                                                          nb.splitter(id=5, order=5), nb.splitter(id=8, order=3),
                                                          nb.splitter(id=7, order=6), nb.splitter(id=3, order=7),
                                                          nb.splitter(id=1, order=8)])

        model = nb.AgglomerativeTree(self.empty_data, 'match_distance')
        match_data = copy.deepcopy(self.data_group)
        result = model.build_hierarchy_tree(match_data)
        self.assertTrue(np.array_equal(march_expected.sum_vector, result.sum_vector),
                        'Agglomerative tree march distance: sum_vector assertion fail')
        self.assertTrue(march_expected.element_id_list == result.element_id_list,
                        'Agglomerative tree march distance: element_id_list assertion fail')
        self.assertTrue((len(march_expected) == len(result)) and
                        all([x == y for x, y in zip(march_expected.splitter_list, result.splitter_list)]),
                        'Agglomerative tree march distance: splitter_list assertion fail')


def suite(test_classes):
    suite_list = []
    loader = unittest.TestLoader()
    for test_class in test_classes:
        print(test_class)
        suite = loader.loadTestsFromTestCase(test_class)
        suite_list.append(suite)
    return suite_list


def main():
    # test list
    mmm_test = [AgglomerativeTreeTest]
    gmm_test = [GMMTest]

    # debug
    require_test = 'MMM'
    # print('Current supported test: [ MMM GMM ]')
    # require_test = input("Test list: ")
    require_test = [x.lower() for x in require_test.split()]
    test_list = []
    if 'mmm' in require_test:
        test_list.extend(mmm_test)
    if 'gmm' in require_test:
        test_list.extend(gmm_test)

    test_suite = unittest.TestSuite(suite(test_list))
    runner = unittest.TextTestRunner()
    runner.run(test_suite)


if __name__ == '__main__':
    main()